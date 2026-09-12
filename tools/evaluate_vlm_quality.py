#!/usr/bin/env python3
"""Check saved VLM answers for basic, image-specific quality regressions."""
import argparse
import json
import re
from pathlib import Path


HAN_OR_KANA = re.compile(r'[\u3400-\u4dbf\u4e00-\u9fff\u3040-\u30ff]')
HANGUL = re.compile(r'[\uac00-\ud7af]')
TOKEN = re.compile(r'[\w가-힣]+', re.UNICODE)


def answers(data, question=None):
    records = data.get('runs', data.get('qa', []))
    for record in records:
        if question is not None and record.get('question') != question:
            continue
        text = record.get('text', record.get('answer', '')).strip()
        yield record, text


def longest_repeat(text):
    tokens = [token.casefold() for token in TOKEN.findall(text)]
    longest = current = 0
    previous = None
    for token in tokens:
        current = current + 1 if token == previous else 1
        longest = max(longest, current)
        previous = token
    return longest


def inspect(text, language, required_groups, max_repeat):
    folded = text.casefold()
    checks = {
        'nonempty': bool(text),
        'no_replacement_character': '\ufffd' not in text,
        'expected_script': (not HAN_OR_KANA.search(text) and
                            (language != 'en' or not HANGUL.search(text))),
        'max_consecutive_token_repeat': longest_repeat(text) <= max_repeat,
        'required_concepts': all(
            any(term.casefold() in folded for term in group)
            for group in required_groups),
    }
    return checks


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('result', type=Path)
    parser.add_argument('--language', choices=('en', 'ko'), required=True)
    parser.add_argument('--question', help='only evaluate this exact saved question')
    parser.add_argument('--image-sha256', help='reject a result from another image')
    parser.add_argument('--require-any', action='append', default=[], metavar='A,B',
                        help='comma-separated synonyms; every group must match')
    parser.add_argument('--max-repeat', type=int, default=2)
    parser.add_argument('--out', type=Path)
    args = parser.parse_args()
    if args.max_repeat < 1:
        parser.error('max-repeat must be positive')
    groups = [[term.strip() for term in value.split(',') if term.strip()]
              for value in args.require_any]
    if any(not group for group in groups):
        parser.error('require-any groups cannot be empty')

    data = json.loads(args.result.read_text())
    image_matches = (not args.image_sha256 or
                     data.get('image_sha256') == args.image_sha256)
    records = []
    for index, (record, text) in enumerate(answers(data, args.question)):
        checks = inspect(text, args.language, groups, args.max_repeat)
        records.append({'index': index, 'kind': record.get('kind'),
                        'text': text, 'checks': checks,
                        'passed': all(checks.values())})
    summary = {
        'schema_version': 1,
        'source': str(args.result),
        'image_sha256': data.get('image_sha256'),
        'expected_image_sha256': args.image_sha256,
        'image_matches': image_matches,
        'language': args.language,
        'required_any': groups,
        'passed': image_matches and bool(records) and
                  all(record['passed'] for record in records),
        'records': records,
    }
    rendered = json.dumps(summary, ensure_ascii=False, indent=2) + '\n'
    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(rendered)
    print(rendered, end='')
    raise SystemExit(0 if summary['passed'] else 3)


if __name__ == '__main__':
    main()
