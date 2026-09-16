"""파이에서 쓰는 STT·TTS 대역 — 실제 계산은 젯슨이 한다.

voice/stt.py, voice/tts.py 와 같은 모양을 갖는다. 그래서 VoicePipeline 은
로컬에서 도는지 원격인지 모른 채 그대로 돌아간다. 상태 기계도 barge-in 도
건드릴 필요가 없다.

TTS 응답은 조각으로 온다. 문장 단위로 합성하는 쪽이 만드는 대로 보내므로,
받는 즉시 재생 큐에 넣으면 첫 소리가 빨리 난다 — 전부 받고 재생하면 그 이점이
사라진다.
"""
import threading

from . import link


class RemoteLink:
    """젯슨과의 연결 하나를 지킨다. 끊기면 다시 붙는다."""

    def __init__(self, host, port):
        self.host, self.port = host, port
        self.sock = None
        self.lock = threading.Lock()

    def ensure(self):
        if self.sock is None:
            self.sock = link.connect(self.host, self.port)
        return self.sock

    def close(self):
        if self.sock:
            try:
                self.sock.close()
            finally:
                self.sock = None

    def reset(self):
        """오류가 나면 소켓을 버린다. 반쯤 읽힌 상태로 재사용하면 규약이 깨진다."""
        self.close()


class RemoteStt:
    name = "원격 STT (젯슨)"

    def __init__(self, conn: RemoteLink):
        self.conn = conn

    def transcribe(self, audio, samplerate=16000):
        if samplerate != link.RATE:
            from .audio import resample
            audio = resample(audio, samplerate, link.RATE)
        with self.conn.lock:
            try:
                s = self.conn.ensure()
                link.send(s, link.STT_REQ, link.pcm_from(audio))
                kind, body = link.recv(s)
            except Exception as e:
                self.conn.reset()
                print(f"  ! STT 요청 실패: {type(e).__name__}: {e}")
                return ""
        if kind == link.STT_RES:
            return body.decode("utf-8", "replace").strip()
        print(f"  ! STT 응답이 이상하다: {kind} {body[:80]!r}")
        return ""


class RemoteTts:
    """젯슨이 합성한 조각을 받아 그대로 재생 큐에 넣는다."""

    samplerate = link.RATE

    def __init__(self, conn: RemoteLink):
        self.conn = conn
        self.providers = ["remote"]
        self.load_s = 0.0

    def speak(self, text, player, stop_event=None, on_first=None):
        # 이미 끊긴 상태면 요청조차 보내지 않는다. 보내고 버리면 젯슨이 GPU 로
        # 헛일을 하고, 그 응답이 소켓에 남아 다음 요청과 섞인다.
        if stop_event is not None and stop_event.is_set():
            return False
        with self.conn.lock:
            try:
                s = self.conn.ensure()
                link.send(s, link.TTS_REQ, text.encode("utf-8"))
                first = True
                while True:
                    if stop_event is not None and stop_event.is_set():
                        # 중간에 끊으면 남은 조각이 소켓에 남는다. 그대로 두면
                        # 다음 요청의 응답과 섞이므로 연결을 버린다.
                        self.conn.reset()
                        return False
                    kind, body = link.recv(s)
                    if kind == link.TTS_AUD:
                        player.put(link.pcm_to(body), link.RATE)
                        if first:
                            first = False
                            if on_first:
                                on_first()
                    elif kind == link.TTS_END:
                        return True
                    else:
                        print(f"  ! TTS 응답이 이상하다: {kind} {body[:80]!r}")
                        self.conn.reset()
                        return False
            except Exception as e:
                self.conn.reset()
                print(f"  ! TTS 요청 실패: {type(e).__name__}: {e}")
                return False
