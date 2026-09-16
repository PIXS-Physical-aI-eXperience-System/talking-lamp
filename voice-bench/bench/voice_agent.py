"""젯슨에서 띄우는 음성 판단부. ROS 노드가 붙어서 오디오를 주고받는다.

    venvs/melo-onnx/bin/python bench/voice_agent.py

모델을 한 번만 올리고 계속 떠 있는다. 적재에 40초 넘게 걸리므로 요청마다
올리는 구조로는 대화가 안 된다.

ROS 를 여기서 import 하지 않는다. 직접 빌드한 onnxruntime·ctranslate2 휠이
ROS 의존성과 부딪히는 것이 가장 깨지기 쉬운 지점이라, 노드는 시스템 파이썬에
두고 이 프로세스와는 localhost 소켓으로만 잇는다.

'생각' 은 우리 파트가 아니다 — VLM 은 A 다. 여기서는 되받아 말하는 것으로
자리만 잡아 둔다.
"""
import argparse
import os
import socket
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, ROOT)

from voice import link                    # noqa: E402
from voice.agent import VoiceAgent        # noqa: E402
from voice.stt import Stt                 # noqa: E402
from voice.tts import Tts                 # noqa: E402
from voice.wake import PHRASE, load_wake  # noqa: E402


class ToneTts:
    """TTS 자리에 끼우는 440Hz 순음.

    파이가 success=True code=drained 를 돌려주는데도 소리가 안 났다. 문서에
    적힌 대로 drained 는 파이프라인이 정상 종료했다는 뜻일 뿐, 스피커에서
    소리가 났다는 보증이 아니다. 우리가 보낸 오디오 내용이 문제인지 전송
    경로가 문제인지 가르려면 확실히 들리는 것을 보내 봐야 한다.
    """

    samplerate = 16000
    providers = ["tone"]
    load_s = 0.0

    def synth(self, text):
        import numpy as np
        n = int(1.0 * self.samplerate)
        t = np.arange(n) / self.samplerate
        # -6 dBFS. 작아서 안 들리는 경우를 배제한다.
        return (0.5 * np.sin(2 * np.pi * 440 * t)).astype(np.float32)


def serve(agent_factory, host, port):
    srv = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    srv.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    srv.bind((host, port))
    srv.listen(1)
    print(f"  대기 중 {host}:{port}\n")
    while True:
        sock, addr = srv.accept()
        sock.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
        print(f"  노드 연결 {addr[0]}:{addr[1]}")
        # 연결마다 판단부를 새로 만든다. 앞 연결의 상태(말하던 중이라거나
        # 발화를 모으던 중이라거나)를 물려받으면 안 된다.
        agent = agent_factory(lambda k, p: _send(sock, k, p))
        try:
            _pump(sock, agent)
        except (ConnectionError, OSError) as e:
            print(f"  연결 끊김: {e}")
        finally:
            sock.close()


def _send(sock, kind, payload):
    try:
        link.send(sock, kind, payload)
    except OSError as e:
        print(f"  ! 노드로 못 보냈다: {e}")


def _pump(sock, agent):
    while True:
        kind, body = link.recv(sock)
        if kind == link.CAP_FRAME:
            sid, pcm = link.unpack_id(body)
            agent.on_capture(sid, link.pcm_to(pcm))
        elif kind == link.ORIENT:
            sid, state = link.unpack_id(body)
            agent.on_orientation(sid, state.decode("utf-8", "replace"))
        elif kind == link.PING:
            link.send(sock, link.PONG)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--port", type=int, default=5150)
    ap.add_argument("--wake-model", default="models/wake/pixs-ya.onnx")
    ap.add_argument("--stt-device", default="cuda", choices=["cuda", "cpu"])
    ap.add_argument("--providers", default="CUDAExecutionProvider,CPUExecutionProvider")
    ap.add_argument("--tone", action="store_true",
                    help="TTS 대신 440Hz 순음을 보낸다. 소리가 안 날 때 "
                         "오디오 내용 문제인지 전송 경로 문제인지 가른다 — "
                         "저쪽 검수에서 440Hz 톤은 들렸다고 기록돼 있다")
    ap.add_argument("--rise-db", type=float, default=12.0,
                    help="재생 중 바닥 대비 몇 dB 오르면 끼어든 것으로 볼지. "
                         "실측에서 끼어들면 최악 28 dB 튀었다")
    args = ap.parse_args()

    print("적재 중…")
    t0 = time.time()
    tts = ToneTts() if args.tone else Tts(providers=args.providers)
    stt = Stt(device=args.stt_device)
    wake_path = os.path.join(ROOT, args.wake_model)
    wake = load_wake(wake_path if os.path.exists(wake_path) else None)
    print(f"  TTS {tts.providers}  {tts.load_s}s"
          + ("   ← 순음 시험 모드" if args.tone else ""))
    print(f"  STT {stt.name}")
    print(f"  웨이크워드 {wake.name}")
    if not getattr(wake, "ready", False):
        print(f'  ! "{PHRASE}" 모델이 없다. 지금은 아무 말에나 깨어난다 — 제품이 아니다.')
    print(f"  합계 {time.time() - t0:.1f}s")

    def think(text):
        print(f"  들은 말: {text}")
        return f"{text}, 라고 하셨네요."

    mark = {"대기": "·", "듣기": "◉", "생각": "…", "말하기": "▶"}
    serve(lambda send: VoiceAgent(stt, tts, wake, think, send,
                                  on_state=lambda s: print(f"  [{mark.get(s,' ')}] {s}"),
                                  rise_db=args.rise_db),
          args.host, args.port)
    return 0


if __name__ == "__main__":
    sys.exit(main())
