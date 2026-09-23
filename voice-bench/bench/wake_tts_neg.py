"""합성 음성으로 부정 데이터를 만든다.

왜: 지금 부정 녹음이 90개뿐이라 "픽스야가 아닌 한국어"를 거의 안 보여줬다.
사람한테 5분씩 더 받는 게 정공법이지만, 잡담을 기다리는 것보다 헷갈릴 만한
소리를 일부러 만들어 넣는 쪽이 발음 조합은 훨씬 촘촘하다.

한계를 먼저 적는다. 저장소의 한국어 TTS는 melo·piper·mimic3 셋 다 KSS
데이터셋이라 **사실상 한 사람 목소리**다. 목소리 다양성은 여기서 못 얻는다.
속도·음높이를 넓게 흔들어 가짜 화자를 만들지만 진짜 사람 다섯 명과 같지 않다.

그래서 효과가 있는지는 **진짜 사람 녹음으로 평가해서** 판단한다.
학습에만 합성을 넣고 평가는 건드리지 않는다(bench/wake_eval.py).

    venvs/melo-onnx/bin/python bench/wake_tts_neg.py
"""
import argparse
import os
import sys
import wave

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, ROOT)

# "픽스야" 와 부딪히는 소리. 가장 위험한 건 [X]스야 꼴이다 —
# 앞 음절만 다르고 뒤가 통째로 같다.
SUEL = [
    "박스", "주스", "가스", "케이스", "마우스", "소스", "뉴스", "버스",
    "테스트", "텍스트", "팩스", "왁스", "체스", "코스", "보스", "캔버스",
    "캠퍼스", "비즈니스", "프로세스", "액세스", "크로스", "드레스", "프레스",
    "스트레스", "플러스", "마이너스", "서비스", "오피스", "페이스", "스페이스",
    "믹스", "식스", "맥스", "엑스", "넥스트", "치즈", "퀴즈", "사이즈",
    "재즈", "피스", "리스트", "포스트", "코스트", "호스트", "게스트",
]
# 픽- 으로 시작하는 말
PIK = ["픽셀", "픽업", "픽션", "픽처", "픽토그램", "피크", "빅스", "믹서",
       "식사", "시스템", "식수", "칙칙", "직선", "씩씩", "픽", "빅", "식",
       "믹", "직", "칙"]

TEMPLATES = [
    "그거 {}야", "이게 {}야?", "{}야 저기 있어", "저건 {}야",
    "{} 좀 가져다줘", "{} 어디 뒀어", "{}는 안 쓸 거야",
]
PIK_TEMPLATES = [
    "{} 좀 봐봐", "{} 어떻게 된 거야", "그 {} 말이야", "{}이 문제인데",
    "{} 쪽으로 가줘",
]

# 평범한 대화. 호출어와 상관없는 말에도 안 깨어나야 한다.
PLAIN = [
    "오늘 날씨가 꽤 춥네", "점심 뭐 먹을까", "회의가 세 시로 밀렸어",
    "그 파일 어디에 저장했지", "배터리가 거의 다 됐는데", "창문 좀 열어줄래",
    "책상 정리를 해야겠다", "커피 한 잔 더 마실까", "생각보다 오래 걸리네",
    "내일 아침에 다시 얘기하자", "이거 어제 끝냈어야 했는데",
    "버스가 십 분이나 늦었어", "지금 몇 시인지 알아", "조금만 기다려 줄래",
    "화면이 너무 어두운 것 같아", "소리가 잘 안 들려", "다시 한번 말해줄래",
    "그렇게 하면 안 될 것 같은데", "괜찮으면 같이 갈까", "우산 챙겨 가야겠다",
    "이번 주는 계속 바빴어", "잠깐 쉬었다 하자", "전화 왔었는데 못 받았어",
    "노트북 충전기 봤어", "여기 앉아도 될까", "생일 선물 뭐 사지",
    "운동을 좀 해야 하는데", "잠이 부족한 것 같아", "약속 시간 늦겠다",
    "이 방 너무 건조하지 않아", "에어컨 온도 좀 낮춰줘", "택배가 아직 안 왔어",
    "숙제 다 했어?", "영화 뭐 볼지 정했어", "밥 먹고 산책할까",
    "그 사람 이름이 뭐였지", "지도 좀 찾아봐줘", "여기서 얼마나 걸려",
    "가격이 생각보다 비싸네", "조명이 좀 눈부신데",
]
# 램프가 실제로 말할 문장. 스피커로 나가는 게 이 목소리라 자기 말에
# 깨어나면 안 된다. 이건 합성으로 넣는 게 오히려 정확하다.
LAMP = [
    "네, 그렇게 할게요", "왼쪽을 더 밝게 했어요", "잘 모르겠어요",
    "조금 더 크게 말씀해 주세요", "밝기를 낮췄습니다", "지금 몇 시인지 알려드릴까요",
    "알겠습니다", "다시 말씀해 주시겠어요", "오른쪽으로 돌렸어요",
    "불을 껐습니다", "무엇을 도와드릴까요", "네, 듣고 있어요",
]


def sentences():
    out = []
    for i, w in enumerate(SUEL):
        for t in TEMPLATES[i % 3::3]:
            out.append(t.format(w))
    for i, w in enumerate(PIK):
        for t in PIK_TEMPLATES[i % 2::2]:
            out.append(t.format(w))
    return out + PLAIN + LAMP


def to16k(x, sr):
    from scipy.signal import resample_poly
    from math import gcd
    if sr == 16000:
        return x
    g = gcd(sr, 16000)
    return resample_poly(x, 16000 // g, sr // g).astype(np.float32)


def write(path, x):
    pcm = (np.clip(x, -1, 1) * 32767).astype("<i2")
    with wave.open(path, "wb") as w:
        w.setnchannels(1); w.setsampwidth(2); w.setframerate(16000)
        w.writeframes(pcm.tobytes())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=os.path.join(ROOT, "out/wake-tts-neg"))
    ap.add_argument("--providers", default="CPUExecutionProvider")
    ap.add_argument("--limit", type=int)
    a = ap.parse_args()

    from voice.tts import Tts
    texts = sentences()
    if a.limit:
        texts = texts[:a.limit]
    os.makedirs(a.out, exist_ok=True)
    with open(os.path.join(a.out, "문장.txt"), "w") as f:
        f.write("\n".join(texts) + "\n")

    t = Tts(providers=a.providers)
    print(f"{len(texts)}문장 합성 — {a.out}")
    for i, text in enumerate(texts):
        path = os.path.join(a.out, f"{i:04d}.wav")
        if os.path.exists(path):
            continue
        try:
            write(path, to16k(t.synth(text), t.samplerate))
        except Exception as e:
            print(f"\n  ! {text!r} 건너뜀 ({type(e).__name__}: {e})")
            continue
        if i % 10 == 0:
            print(f"\r  {i+1}/{len(texts)}", end="", flush=True)
    print(f"\r  {len(texts)}/{len(texts)} 끝")


if __name__ == "__main__":
    main()
