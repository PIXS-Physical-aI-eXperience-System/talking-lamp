"""파이에서 돌리는 음성 에이전트.

마이크·스피커·웨이크워드·VAD 는 파이가 맡고, STT·TTS 는 젯슨에 맡긴다.
상태 기계(VoicePipeline)는 젯슨용과 같은 것을 그대로 쓴다 — STT·TTS 자리에
원격 대역을 끼울 뿐이다.

    python bench/pi_agent.py --jetson 192.168.0.10

barge-in 은 파이가 혼자 처리한다. 사용자가 끼어들면 네트워크를 타지 않고
그 자리에서 재생을 멈춘다. 지연에 가장 민감한 구간에서 LAN 을 빼는 것이
이 구조의 핵심이다.
"""
import argparse
import os
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, ROOT)

from voice import VoicePipeline                      # noqa: E402
from voice.remote import RemoteLink, RemoteStt, RemoteTts   # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--jetson", required=True, help="젯슨 주소")
    ap.add_argument("--port", type=int, default=5150)
    ap.add_argument("--wake-model", default="models/wake/pixs-ya.onnx")
    ap.add_argument("--say", help="시작하자마자 이 문장을 말한다")
    args = ap.parse_args()

    from voice.audio import find_device
    for kind, why in (("input", "마이크"), ("output", "스피커(하드웨어 AEC 의 전제)")):
        try:
            idx, d = find_device(kind)
        except OSError as e:
            print(f"오디오를 쓸 수 없다: {e}")
            return 1
        if idx is None:
            print(f"XVF3800 을 {why}로 찾지 못했다. bench/mic_check.py 를 먼저 실행할 것")
            return 1
        print(f"  {why}: [{idx}] {d['name']} @{int(d['default_samplerate'])}Hz")

    conn = RemoteLink(args.jetson, args.port)
    try:
        t0 = time.perf_counter()
        conn.ensure()
        print(f"  젯슨 {args.jetson}:{args.port} 연결 {(time.perf_counter()-t0)*1000:.0f}ms")
    except Exception as e:
        print(f"젯슨에 붙지 못했다: {e}")
        print("  bench/voice_server.py 가 젯슨에서 떠 있는지 확인할 것")
        return 1

    def think(text):
        print(f"  들은 말: {text}")
        return f"{text}, 라고 하셨네요."

    mark = {"대기": "·", "듣기": "◉", "생각": "…", "말하기": "▶"}
    pipe = VoicePipeline(
        RemoteStt(conn), RemoteTts(conn),
        wake_model=args.wake_model if os.path.exists(args.wake_model) else None,
        on_utterance=think,
        on_state=lambda s: print(f"  [{mark.get(s, ' ')}] {s}"))

    try:
        pipe.start()
    except Exception as e:
        print(f"\n시작하지 못했다: {e}")
        return 1
    if args.say:
        pipe.say(args.say)
    print('  준비됐다. "픽스야" 라고 부른 뒤 말해 보라. Ctrl+C 로 종료.\n')
    try:
        while True:
            time.sleep(0.5)
    except KeyboardInterrupt:
        print("\n종료 중…")
        pipe.stop()
        conn.close()
    return 0


if __name__ == "__main__":
    sys.exit(main())
