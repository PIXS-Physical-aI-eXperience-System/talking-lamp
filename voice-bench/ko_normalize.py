"""한국어 TTS 전처리 — 숫자·영문을 '읽는 대로' 한글로 바꾼다.

어떤 TTS 엔진을 고르든 필요한 계층이다. 엔진은 글자를 소리로 바꿀 뿐,
'15도'를 '십오 도'로 읽어야 하는지 '열다섯 도'로 읽어야 하는지는 모른다.

한국어 수사는 단위에 따라 갈린다:
    3시  → 세 시   (고유어)
    3분  → 삼 분   (한자어)
    15도 → 십오 도 (한자어)
그래서 단위를 보고 어느 쪽을 쓸지 결정해야 한다.
"""
import re

_SINO_D = "영일이삼사오육칠팔구"
_SINO_U = ["", "십", "백", "천"]

_NATIVE_ONES = ["", "하나", "둘", "셋", "넷", "다섯", "여섯", "일곱", "여덟", "아홉"]
_NATIVE_TENS = ["", "열", "스물", "서른", "마흔", "쉰", "예순", "일흔", "여든", "아흔"]
# 단위 앞에서 형태가 바뀌는 관형사형: 하나→한, 둘→두 …
_ATTRIB = {"하나": "한", "둘": "두", "셋": "세", "넷": "네", "스물": "스무"}

# 고유어로 세는 단위. 나머지는 한자어로 본다.
_NATIVE_COUNTERS = {
    "시", "개", "명", "살", "마리", "번", "잔", "그루", "송이", "권", "장",
    "대", "켤레", "벌", "채", "군데", "가지", "돌", "숟갈", "봉지",
}

_ALPHA = {
    "A": "에이", "B": "비", "C": "씨", "D": "디", "E": "이", "F": "에프",
    "G": "지", "H": "에이치", "I": "아이", "J": "제이", "K": "케이",
    "L": "엘", "M": "엠", "N": "엔", "O": "오", "P": "피", "Q": "큐",
    "R": "아르", "S": "에스", "T": "티", "U": "유", "V": "브이",
    "W": "더블유", "X": "엑스", "Y": "와이", "Z": "지",
}

# 통째로 읽는 게 자연스러운 낱말은 철자 읽기보다 우선한다.
_WORDS = {
    "USB": "유에스비", "AI": "에이아이", "TV": "티비", "PC": "피씨",
    "LED": "엘이디", "CPU": "씨피유", "GPU": "지피유", "OK": "오케이",
    "WIFI": "와이파이", "WI-FI": "와이파이", "HDMI": "에이치디엠아이",
}


def sino(n: int) -> str:
    """한자어 수사: 15 → 십오"""
    if n == 0:
        return "영"
    out = []
    for unit_idx, chunk_start in ((1, 10000),):  # 만 단위
        if n >= chunk_start:
            out.append(sino(n // chunk_start) if n // chunk_start != 1 else "")
            out.append("만")
            n %= chunk_start
    s = ""
    digits = str(n)
    for i, ch in enumerate(digits):
        d = int(ch)
        pos = len(digits) - i - 1
        if d == 0:
            continue
        # 십/백/천 앞의 1은 읽지 않는다 (십오 ○, 일십오 ✗)
        s += ("" if d == 1 and pos > 0 else _SINO_D[d]) + _SINO_U[pos]
    return "".join(out) + s


def native(n: int, attributive: bool = False) -> str:
    """고유어 수사: 3 → 셋, 단위 앞이면 세"""
    if not 1 <= n <= 99:
        return sino(n)  # 고유어는 99까지만 자연스럽다
    s = _NATIVE_TENS[n // 10] + _NATIVE_ONES[n % 10]
    if attributive:
        for full, short in _ATTRIB.items():
            if s.endswith(full):
                return s[: -len(full)] + short
    return s


def _num_with_counter(m: re.Match) -> str:
    """숫자 뒤 한글 덩어리에서 단위를 떼어낸다.

    '3시에' 처럼 단위 뒤에 조사가 붙으므로 덩어리 전체를 단위로 보면 안 된다.
    고유어 단위로 시작하는지를 앞에서부터 확인한다 ('시에' → 단위 '시' + 조사 '에').
    """
    n, tail = int(m.group(1)), m.group(2)
    for c in sorted(_NATIVE_COUNTERS, key=len, reverse=True):
        if tail.startswith(c):
            return f"{native(n, attributive=True)} {tail}"
    return f"{sino(n)} {tail}"


def normalize(text: str) -> str:
    """TTS에 넣기 전 텍스트를 한글 읽기로 바꾼다."""
    # 1) 통짜로 읽는 낱말 (USB → 유에스비)
    for word, reading in sorted(_WORDS.items(), key=lambda kv: -len(kv[0])):
        text = re.sub(rf"\b{re.escape(word)}\b", reading, text, flags=re.IGNORECASE)

    # 2) 숫자 + 단위 (단위를 보고 고유어/한자어를 고른다)
    text = re.sub(r"(\d+)\s*([가-힣]+)", _num_with_counter, text)

    # 3) 남은 맨숫자는 한자어로
    text = re.sub(r"\d+", lambda m: sino(int(m.group())), text)

    # 4) 남은 알파벳은 한 글자씩 (KBS → 케이비에스)
    text = re.sub(r"[A-Za-z]+",
                  lambda m: "".join(_ALPHA.get(c.upper(), c) for c in m.group()), text)
    return text


if __name__ == "__main__":
    for t in ["눈부신데 각도 15도만 내려줄래?", "승원아, 3시에 회의 있어.",
              "USB 케이블이랑 노트북 좀 찾아줘.", "3분 뒤에 알려줘.",
              "커피 2잔 시켰어.", "온도를 23도로 맞춰줘.", "LED 5개 켜줘."]:
        print(f"{t:<28} → {normalize(t)}")
