"""모인 웨이크워드 녹음을 읽는다. 폴더 구조가 사람마다 다르다.

우리 도구(bench/wake_record.py)는 이렇게 만든다:

    <이름>/pos/*.wav   <이름>/neg/*.wav   <이름>/noise.wav

그런데 각자 자기 도구로 녹음해 오기도 한다. 실제로 받은 것:

    <이름>/wake/*.wav  <이름>/hard/*.wav  <이름>/sentences/*.wav
    <이름>/noise/room.wav  + manifest.jsonl, meta.json

내용은 같다(호출어 20개, 헷갈리는 말 12개, 문장 6개, 방 소리 1개, 16 kHz).
파일을 옮겨 규격을 맞추게 하지 않는다 — 보낸 사람의 것을 손대는 셈이고,
다음 사람이 또 다른 구조로 보내면 같은 일을 반복한다. 읽는 쪽에서 흡수한다.
"""
import glob
import os

# (긍정으로 볼 폴더, 부정으로 볼 폴더들, 잡음 찾을 자리)
LAYOUTS = (
    ("pos", ("neg",), ("noise.wav",)),
    ("wake", ("hard", "sentences"), ("noise/*.wav", "noise.wav")),
)


def load_person(person_dir):
    """(긍정 목록, 부정 목록, 잡음 목록). 구조를 못 알아보면 전부 빈 목록."""
    for pos_dir, neg_dirs, noise_pats in LAYOUTS:
        pos = sorted(glob.glob(os.path.join(person_dir, pos_dir, "*.wav")))
        if not pos:
            continue
        neg = []
        for d in neg_dirs:
            neg += sorted(glob.glob(os.path.join(person_dir, d, "*.wav")))
        noise = []
        for pat in noise_pats:
            noise += sorted(glob.glob(os.path.join(person_dir, pat)))
        return pos, sorted(neg), noise
    return [], [], []


def people(base):
    """[(이름, 긍정, 부정, 잡음)]. 이름순."""
    out = []
    for d in sorted(glob.glob(os.path.join(base, "*"))):
        if not os.path.isdir(d):
            continue
        pos, neg, noise = load_person(d)
        out.append((os.path.basename(d), pos, neg, noise))
    return out
