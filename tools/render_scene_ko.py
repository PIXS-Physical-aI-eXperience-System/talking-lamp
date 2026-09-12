#!/usr/bin/env python3
"""Render a constrained English VLM object list as natural Korean."""
import argparse
import json
import re
from pathlib import Path


LABELS = {
    'usb charger': 'USB 충전기',
    'power adapter': '전원 어댑터',
    'power strip': '멀티탭',
    'office chair': '사무용 의자',
    'chair': '의자',
    'desk with items': '물건이 놓인 책상',
    'desk': '책상',
    'table': '테이블',
    'workstation': '작업대',
    'robot arm': '로봇 팔',
    'robotic arm': '로봇 팔',
    'laptop': '노트북',
    'notebook': '공책',
    'notepad': '메모장',
    'keyboard': '키보드',
    'book': '책',
    'cup': '컵',
    'mug': '머그컵',
    'hand': '손',
    'person': '사람',
    'face': '얼굴',
    'circuit board': '회로 기판',
    'blue box with arduino': '아두이노가 든 파란 상자',
    'arduino': '아두이노',
}


def normalize_item(item):
    return re.sub(r'\s+', ' ', item.strip().strip('.').casefold())


def has_batchim(text):
    for char in reversed(text):
        if '가' <= char <= '힣':
            return (ord(char) - ord('가')) % 28 != 0
    return False


def join_korean(items):
    if len(items) == 1:
        return items[0]
    conjunction = '과' if has_batchim(items[-2]) else '와'
    return ', '.join(items[:-2] + [f'{items[-2]}{conjunction} {items[-1]}'])


def render(source_text):
    items = [normalize_item(item) for item in source_text.split(',')]
    items = [item for item in items if item]
    translated = []
    unknown = []
    for item in items:
        value = LABELS.get(item)
        if value is None:
            unknown.append(item)
        elif value not in translated:
            translated.append(value)
    if translated:
        subject = '이' if has_batchim(translated[-1]) else '가'
        text = f"여기에는 {join_korean(translated)}{subject} 보여요."
    else:
        text = ''
    return text, unknown


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--input', type=Path, required=True)
    parser.add_argument('--out', type=Path, required=True)
    args = parser.parse_args()
    source = json.loads(args.input.read_text())
    source_text = source['runs'][0]['text']
    text, unknown = render(source_text)
    result = {
        'schema_version': 1,
        'source': str(args.input),
        'source_text': source_text,
        'language': 'ko',
        'text': text,
        'unknown_items': unknown,
        'passed': bool(text) and not unknown,
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(result, ensure_ascii=False, indent=2) + '\n')
    print(json.dumps(result, ensure_ascii=False), flush=True)
    raise SystemExit(0 if result['passed'] else 3)


if __name__ == '__main__':
    main()
