"""오디오 입출력 — XVF3800 을 찾고, 입력을 모으고, 출력을 끊을 수 있게 재생한다.

두 가지가 이 파일의 이유다.

1) 출력은 반드시 XVF3800 으로 내보내야 한다. 스피커가 그 보드에 물려 있어야
   하드웨어 AEC 가 자기 목소리를 기준으로 삼을 수 있다. 다른 장치로 내보내면
   클럭 도메인이 달라 AEC 가 동작하지 않고, 램프가 제 목소리에 반응한다.

2) 재생은 언제든 즉시 멈출 수 있어야 한다. barge-in 은 사용자가 말을 끊고
   들어오는 순간 TTS 가 멎어야 성립한다. 파일을 통째로 재생하는 방식으로는
   안 되고, 조각을 큐로 물려 놓고 중간에 버릴 수 있어야 한다.
"""
import queue
import threading

import numpy as np

XVF_KEYS = ("xvf", "respeaker", "xmos")


def find_device(kind="input"):
    """XVF3800 의 장치 번호를 찾는다. 없으면 None."""
    import sounddevice as sd
    want = "max_input_channels" if kind == "input" else "max_output_channels"
    for i, d in enumerate(sd.query_devices()):
        if d[want] > 0 and any(k in d["name"].lower() for k in XVF_KEYS):
            return i, d
    return None, None


class Mic:
    """XVF3800 입력을 열어 프레임 단위로 넘겨준다.

    6채널 펌웨어에서 채널 구성은 이렇다:
      0 처리음(회의용)  1 처리음(음성인식용)  2~5 마이크 원음
    웨이크워드·VAD·STT 는 전부 1번(음성인식용)을 쓴다. 원음은 우리가 직접
    빔포밍할 때나 필요한데, 지금은 펌웨어가 해주는 것을 쓴다.
    """

    ASR_CH = 1

    def __init__(self, blocksize=512, samplerate=16000):
        self.blocksize = blocksize
        self.samplerate = samplerate
        self.q = queue.Queue(maxsize=50)
        self.stream = None
        self.channels = 1

    def __enter__(self):
        import sounddevice as sd
        idx, d = find_device("input")
        if idx is None:
            raise RuntimeError("XVF3800 입력 장치를 못 찾았다. bench/mic_check.py 먼저 실행할 것")
        self.channels = int(d["max_input_channels"])
        self.samplerate = int(d["default_samplerate"])

        def cb(indata, frames, t, status):
            # 콜백에서 무거운 일을 하면 오디오가 끊긴다. 복사해서 큐에만 넣는다.
            ch = self.ASR_CH if self.channels > self.ASR_CH else 0
            try:
                self.q.put_nowait(indata[:, ch].copy())
            except queue.Full:
                pass    # 소비가 밀리면 오래된 것부터 버린다. 지연이 쌓이는 것보다 낫다

        self.stream = sd.InputStream(device=idx, channels=self.channels,
                                     samplerate=self.samplerate, blocksize=self.blocksize,
                                     dtype="float32", callback=cb)
        self.stream.start()
        return self

    def __exit__(self, *a):
        if self.stream:
            self.stream.stop()
            self.stream.close()

    def frames(self, timeout=1.0):
        """프레임을 계속 내놓는 제너레이터."""
        while True:
            try:
                yield self.q.get(timeout=timeout)
            except queue.Empty:
                return

    def drain(self):
        """쌓인 프레임을 버린다. 상태가 바뀔 때 과거 소리를 물고 가지 않게."""
        while not self.q.empty():
            try:
                self.q.get_nowait()
            except queue.Empty:
                break


def resample(x, src, dst):
    """표본율 변환. melo 는 44.1 kHz 로 만드는데 보드 출력은 16 kHz 다.

    보드로 내보내야 하드웨어 AEC 가 사는데, 그러려면 내려 깎아야 한다.
    음질 손실이 생기며 그 영향은 아직 측정하지 않았다.
    """
    if src == dst:
        return x
    try:
        from scipy.signal import resample_poly
        from math import gcd
        g = gcd(int(src), int(dst))
        return resample_poly(x, dst // g, src // g).astype(np.float32)
    except ImportError:
        # scipy 가 없으면 선형 보간. 거칠지만 돌아는 간다.
        n = int(round(len(x) * dst / src))
        return np.interp(np.linspace(0, len(x) - 1, n),
                         np.arange(len(x)), x).astype(np.float32)


class Player:
    """조각을 큐로 받아 재생하고, 언제든 즉시 멈춘다."""

    def __init__(self, samplerate=16000):
        self.samplerate = samplerate
        self.q = queue.Queue()
        self.stop_flag = threading.Event()
        self.done = threading.Event()
        self.done.set()
        self.thread = None
        self.device = None

    def open(self):
        idx, d = find_device("output")
        if idx is None:
            raise RuntimeError("XVF3800 출력 장치를 못 찾았다 — 하드웨어 AEC 가 죽는다")
        self.device = idx
        self.samplerate = int(d["default_samplerate"])
        return self

    def put(self, audio, src_rate):
        self.q.put(resample(np.asarray(audio, dtype=np.float32), src_rate, self.samplerate))
        if self.thread is None or not self.thread.is_alive():
            self.stop_flag.clear()
            self.done.clear()
            self.thread = threading.Thread(target=self._run, daemon=True)
            self.thread.start()

    def _run(self):
        import sounddevice as sd
        with sd.OutputStream(device=self.device, channels=1,
                             samplerate=self.samplerate, dtype="float32") as out:
            while not self.stop_flag.is_set():
                try:
                    chunk = self.q.get(timeout=0.2)
                except queue.Empty:
                    break
                # 통째로 write 하면 중간에 못 끊는다. 잘게 나눠 내보내며
                # 매번 정지 신호를 본다 — barge-in 반응 시간이 이 크기로 정해진다.
                step = max(256, self.samplerate // 50)   # 20 ms
                for i in range(0, len(chunk), step):
                    if self.stop_flag.is_set():
                        return
                    out.write(chunk[i:i + step])
        self.done.set()

    def stop(self):
        """즉시 정지. 큐에 남은 조각도 버린다."""
        self.stop_flag.set()
        while not self.q.empty():
            try:
                self.q.get_nowait()
            except queue.Empty:
                break
        self.done.set()

    def is_playing(self):
        return self.thread is not None and self.thread.is_alive() and not self.done.is_set()
