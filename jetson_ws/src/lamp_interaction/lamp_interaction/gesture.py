"""Pick a body gesture from what the user said.

The on-board 1.5B model answers worse when it also has to pick a motion tag:
with a "[nod]"-style prefix it forgot its own name and said there was no cup
the camera saw, and called "나 오늘 너무 피곤해" excited (2026-11-01). So the
model only talks, and the gesture comes from the user's words here. It runs
on the transcript before the reply is spoken, so it costs no latency.
"""

from __future__ import annotations

import re

# First match wins: praise before thanks, so "잘했어 고마워" wiggles.
# Every gesture plays at full size: at 0.5-0.6 the user found them too timid
# (2026-11-02). 1.0 is the ceiling the motion protocol accepts.
_RULES = (
    # Affection before agreement: "좋아해" is not "좋아".
    ("shy", 1.0, re.compile(r"사랑해|좋아해|보고\s*싶었|보고\s*싶어")),
    # The clip is small (wrist about 12 deg at 1.0), so it plays at full size.
    ("shock", 1.0, re.compile(r"깜짝|헐|말도\s*안\s*돼|세상에|어머(?!니)|으악|무서워|놀랐")),
    ("happy_wiggle", 1.0, re.compile(
        r"잘했|잘한다|똑똑|최고|귀엽|귀여워|예쁘|이쁘|멋지|멋있|대단|착하|천재|훌륭|짱")),
    ("excited", 1.0, re.compile(r"합격|축하|성공|이겼|상\s*받|대박|신난|신나|기뻐|좋은\s*소식")),
    ("sad", 1.0, re.compile(r"피곤|우울|슬퍼|슬프|힘들|망쳤|속상|짜증|외로|아파|아프")),
    ("headshake", 1.0, re.compile(r"^\s*(?:아니|싫어|별로|틀렸)|아니야|싫어|틀렸어")),
    ("nod", 1.0, re.compile(
        r"^\s*(?:응|어|네|그래|좋아|맞아|오케이|알았어|알겠어)(?:\s|[.!?,~]|$)"
        r"|고마워|고맙|감사|그렇게\s*하자")),
    # Questions last: tilt the head while thinking about the answer.
    ("curious", 1.0, re.compile(r"궁금|뭘까|맞혀|맞춰|누구야|뭐\s*있|뭐야|뭐\s*보여|어디")),
)

# While the lamp talks with no other gesture this turn: one small nod.
SPEAK_NOD = ("nod", 0.5)


def gesture_for(text: str) -> tuple[str, float] | None:
    for tag, intensity, pattern in _RULES:
        if pattern.search(text):
            return tag, intensity
    return None
