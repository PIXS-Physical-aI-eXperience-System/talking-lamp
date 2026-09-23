"""영어 TTS 대조군. 한국어에서 겪은 품질 한계가 언어 문제인지 확인한다."""
import os, subprocess, sys, time
import numpy as np, soundfile as sf
HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)   # voice-bench/ — 공용 모듈과 산출물이 있는 곳; sys.path.insert(0, ROOT)
from common import load_sentences

def rss():
    return int(subprocess.run(["ps","-o","rss=","-p",str(os.getpid())],
           capture_output=True,text=True).stdout)/1024

from kokoro import KPipeline
sents = load_sentences(os.path.join(HERE, "sentences_en.txt"))
pipe = KPipeline(lang_code="a")   # American English
for voice in ("af_heart", "am_michael"):
    out = os.path.join(ROOT, "out", "tts", f"kokoro-en-{voice}")
    os.makedirs(out, exist_ok=True)
    syn = aud = 0.0
    for i, s in enumerate(sents, 1):
        t0 = time.perf_counter()
        chunks = [a for _g, _p, a in pipe(s, voice=voice)]
        a = np.concatenate(chunks) if len(chunks) > 1 else chunks[0]
        syn += time.perf_counter() - t0; aud += len(a) / 24000
        sf.write(os.path.join(out, f"{i:02d}.wav"), a, 24000)
    print(f"{voice:<14} RTF {syn/aud:.3f}   피크 RSS {rss():.0f} MB")
