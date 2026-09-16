"""TTS — 문장 단위로 쪼개 합성하고, 만들자마자 재생한다.

통째로 합성하면 두 가지가 나빠진다. Jetson 실측(138자, 오디오 27초):

              통째로      문장 단위 9조각
  최고 메모리  1750 MB     1245 MB      (-505 MB)
  말 시작까지  4.56s       0.59s        (7.7배)
  전체 합성    4.56s       7.22s

전체 합성 시간이 느는 것은 문제가 안 된다. 첫 조각을 재생하는 동안 다음을
만들면 되고, RTF 0.17 이라 합성이 재생을 계속 앞지른다.

메모리가 뛰는 이유는 VITS 가 파형 전체를 한 번에 만들기 때문이다. 긴 출력이면
중간 텐서가 그만큼 커진다.

구성은 VITS fp32 + BERT int8 + CUDA 다. int8 을 VITS 까지 내리면 GPU 에서
10.4배 느려진다(RTF 2.61 vs 0.250) — 동적 양자화가 CPU 커널용이라 대응 커널이
없는 노드가 CPU 로 폴백하고 그 경계마다 변환이 붙는다.
"""
import os
import re
import sys
import threading

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(HERE, "runners"))

# 문장 끝에서 자르되 문장부호는 남긴다. 구분자에 마침표를 포함시키면 잘려
# 나가서 "아직 작업 중이시네요" 처럼 끝맺음이 사라지고 억양이 달라진다.
# 숫자 사이의 마침표(3.5)는 뒤에 한글/영문이 와야 자르므로 걸리지 않는다.
_SPLIT = re.compile(r'(?<=[.!?。？！])\s+|(?<=[.!?。？！])(?=[가-힣A-Za-z])')


def split_sentences(text, max_chars=60):
    """문장 단위로 쪼갠다. 한 조각이 너무 길면 쉼표에서 더 자른다.

    조각 길이가 메모리 최고점을 정하므로 상한을 둔다. 맥/Jetson 실측에서
    75자(오디오 15초)까지는 평평했고 그 위에서 뛰었다.
    """
    parts = [p.strip() for p in _SPLIT.split(text) if p and p.strip()]
    out = []
    for p in parts:
        while len(p) > max_chars:
            cut = p.rfind(",", 0, max_chars)
            if cut < max_chars // 3:
                cut = max_chars
            out.append(p[:cut + 1].strip())
            p = p[cut + 1:].strip()
        if p:
            out.append(p)
    return out or [text]


class Tts:
    def __init__(self, model_dir="models/melo-ko-onnx",
                 providers="CUDAExecutionProvider,CPUExecutionProvider",
                 threads=2, normalize=True):
        from tts_melo_onnx import build_synth
        self._synth, self.samplerate, self.providers, self.load_s, _ = build_synth(
            model_dir, int8=False, bert_int8=True,      # VITS fp32 + BERT int8
            providers=providers, threads=threads, quiet=True)
        self.normalize = normalize
        self._norm = None
        if normalize:
            from ko_normalize import normalize as n
            self._norm = n
        self._lock = threading.Lock()

    def synth(self, text):
        """한 조각을 합성한다. 세션은 스레드 안전하지 않으므로 잠근다."""
        with self._lock:
            return self._synth(self._norm(text) if self._norm else text)

    def speak(self, text, player, stop_event=None, on_first=None):
        """쪼개서 합성하고 그때그때 재생 큐에 넣는다.

        stop_event 가 서면 남은 조각을 만들지 않는다. barge-in 때 이미 만든
        것만 버리면 되는 게 아니라, 앞으로 만들 것도 멈춰야 GPU 가 놀지 않는다.
        """
        first = True
        for part in split_sentences(text):
            if stop_event is not None and stop_event.is_set():
                return False
            audio = self.synth(part)
            if stop_event is not None and stop_event.is_set():
                return False
            player.put(audio, self.samplerate)
            if first:
                first = False
                if on_first:
                    on_first()
        return True
