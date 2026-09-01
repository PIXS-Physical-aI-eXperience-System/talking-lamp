"""스레드 수에 따른 melo 피크 RSS와 속도(RTF)의 교환을 잰다."""
import os, subprocess, sys, tempfile, time
HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)   # voice-bench/ — 공용 모듈과 산출물이 있는 곳; sys.path.insert(0, ROOT)
from common import load_sentences, repo_paths
import torch
N = int(sys.argv[1]); CYCLES = int(sys.argv[2])
torch.set_num_threads(N)
from melo.api import TTS
import soundfile as sf

def rss():
    return int(subprocess.run(["ps","-o","rss=","-p",str(os.getpid())],
           capture_output=True,text=True).stdout.strip())/1024

tts = TTS(language="KR", device="cpu"); spk = tts.hps.data.spk2id["KR"]
tmp = os.path.join(tempfile.mkdtemp(), "o.wav")
sents = load_sentences(repo_paths()["sentences"])
peak, syn, aud = 0.0, 0.0, 0.0
for i in range(CYCLES):
    t0 = time.perf_counter()
    tts.tts_to_file(sents[i % len(sents)], spk, tmp)
    syn += time.perf_counter() - t0
    aud += sf.info(tmp).duration
    peak = max(peak, rss())
print(f"RESULT\t{N}\t{peak:.0f}\t{syn/aud:.3f}")
