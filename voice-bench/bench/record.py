"""심사 문장 6개를 직접 녹음해서 ref/ 에 저장한다.

STT 정확도는 '내 목소리, 내 마이크, 내 환경'에서 재야 의미가 있다.
공개 데이터셋 WER은 참고치일 뿐이고, 실제로 램프한테 말할 때
어떻게 나오는지가 우리가 알아야 할 숫자다.

    python3 record.py            # 전부 녹음
    python3 record.py 3          # 3번 문장만 다시 녹음

의존성: sounddevice, soundfile, numpy
"""
import os
import sys

import numpy as np
import sounddevice as sd
import soundfile as sf

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from common import load_sentences, repo_paths

SAMPLE_RATE = 16000  # STT 후보들이 전부 16k 모노를 기대한다
CHANNELS = 1


def record_one(idx: int, text: str, out_path: str) -> None:
    print(f"\n[{idx}] {text}")
    input("    Enter → 녹음 시작 ")

    frames = []
    with sd.InputStream(samplerate=SAMPLE_RATE, channels=CHANNELS, dtype="float32",
                        callback=lambda ind, *_: frames.append(ind.copy())):
        input("    녹음 중… Enter → 정지 ")

    if not frames:
        print("    ! 녹음된 게 없다. 건너뜀")
        return

    audio = np.concatenate(frames, axis=0)
    sf.write(out_path, audio, SAMPLE_RATE, subtype="PCM_16")
    dur = len(audio) / SAMPLE_RATE
    peak = float(np.abs(audio).max())
    print(f"    저장: {out_path}  ({dur:.1f}초, 피크 {peak:.2f})")
    if peak > 0.99:
        print("    ! 클리핑. 마이크에서 좀 떨어져서 다시 녹음할 것")
    elif peak < 0.05:
        print("    ! 너무 작다. 가까이서 다시 녹음할 것")


def main() -> int:
    p = repo_paths()
    sentences = load_sentences(p["sentences"])

    only = None
    if len(sys.argv) > 1:
        only = int(sys.argv[1])
        if not 1 <= only <= len(sentences):
            print(f"문장 번호는 1~{len(sentences)}")
            return 1

    print(f"입력 장치: {sd.query_devices(kind='input')['name']}")
    print(f"{SAMPLE_RATE} Hz / 모노 / PCM16")
    print("\n※ 실제로 램프한테 말하듯이. 1 m 정도 떨어져서, 평소 말투로.")

    for i, text in enumerate(sentences, 1):
        if only and i != only:
            continue
        record_one(i, text, f"{p['ref']}/{i:02d}.wav")

    print("\n완료. 다음: python3 run.py --stt")
    return 0


if __name__ == "__main__":
    sys.exit(main())
