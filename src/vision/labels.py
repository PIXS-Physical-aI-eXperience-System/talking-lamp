"""COCO class names and their Korean names for the cognition prompt.

D owns this table because the detector decides which classes exist. If the
detector model changes, the label set changes with it, and keeping the mapping
here means A's cognition code does not break when that happens.
"""

from __future__ import annotations

COCO: tuple[str, ...] = tuple(
    "person bicycle car motorcycle airplane bus train truck boat traffic_light "
    "fire_hydrant stop_sign parking_meter bench bird cat dog horse sheep cow "
    "elephant bear zebra giraffe backpack umbrella handbag tie suitcase frisbee "
    "skis snowboard sports_ball kite baseball_bat baseball_glove skateboard "
    "surfboard tennis_racket bottle wine_glass cup fork knife spoon bowl banana "
    "apple sandwich orange broccoli carrot hot_dog pizza donut cake chair couch "
    "potted_plant bed dining_table toilet tv laptop mouse remote keyboard "
    "cell_phone microwave oven toaster sink refrigerator book clock vase "
    "scissors teddy_bear hair_drier toothbrush".split()
)

# Things that sit on a desk. Used for the scene description and live preview.
# "person" is deliberately excluded: people are handled by the face path.
DESK: frozenset[str] = frozenset({
    "book", "keyboard", "mouse", "laptop", "cell_phone", "cup", "bottle",
    "scissors", "remote", "bowl", "clock", "vase",
})

# What S1 may actually point the task light at: things you read or type on.
# Narrower than DESK on purpose. On the head camera (2026-09-28, focused) the
# detector named a flat notepad "keyboard" with a well-placed box, while the
# misreads it made elsewhere were all outside this set: headphones -> "mouse",
# a pen -> "remote"/"cell_phone", a power brick -> "chair". The light should
# follow the notepad and ignore the rest, and this set does exactly that.
TASK_LIGHT: frozenset[str] = frozenset({"book", "laptop", "keyboard"})

# Korean names for the cognition prompt. Taken over from A's
# src/cognition/vision_to_cognition.py (feat/cognition-llm) unchanged, so the
# prompt A tuned against keeps seeing the same words.
KO: dict[str, str] = {
    "person": "사람", "laptop": "노트북", "cup": "컵", "bottle": "물병",
    "book": "책", "keyboard": "키보드", "mouse": "마우스", "cell_phone": "휴대폰",
    "remote": "리모컨", "bowl": "그릇", "clock": "시계", "vase": "꽃병",
    "scissors": "가위", "chair": "의자", "tv": "모니터", "wine_glass": "와인잔",
    "potted_plant": "화분", "teddy_bear": "인형", "dining_table": "책상",
    "fork": "포크", "knife": "칼", "spoon": "숟가락", "banana": "바나나",
    "apple": "사과", "backpack": "가방", "handbag": "가방",
}


def to_korean(names, *, dedupe: bool = True) -> list[str]:
    """COCO names -> Korean names, dropping classes without a Korean entry.

    Unmapped classes are dropped rather than passed through in English: the
    cognition prompt is Korean-only and C's TTS cannot read mixed scripts.
    """
    out: list[str] = []
    for n in names:
        ko = KO.get(n)
        if ko is None or (dedupe and ko in out):
            continue
        out.append(ko)
    return out
