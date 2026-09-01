"""TTS 블라인드 청취.

TTS는 숫자로 못 고른다. 문서에도 '귀로 듣고 정할 것'이라고 적혀 있다.
다만 어느 게 뭔지 알고 들으면 편향이 생기므로, 순서를 섞어서 들려주고
다 들은 뒤에 정답을 공개한다.

    python3 blind.py            # 전 문장
    python3 blind.py --sentence 4

점수는 out/blind_scores.json 에 누적된다. 팀원 여러 명이 각자 돌리고
--rater 로 이름을 남기면 나중에 합산할 수 있다.
"""
import argparse
import glob
import json
import os
import random
import subprocess
import sys
from collections import defaultdict

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from common import load_sentences, repo_paths

PLAYERS = [["afplay"], ["aplay", "-q"], ["ffplay", "-nodisp", "-autoexit", "-loglevel", "quiet"]]


def find_player():
    for p in PLAYERS:
        if subprocess.run(["which", p[0]], capture_output=True).returncode == 0:
            return p
    return None


def collect_wavs(out_dir: str) -> dict[int, list[tuple[str, str]]]:
    """{문장번호: [(후보라벨, wav경로), ...]}"""
    by_sentence = defaultdict(list)
    for cand_dir in sorted(glob.glob(os.path.join(out_dir, "tts", "*"))):
        if not os.path.isdir(cand_dir):
            continue
        label = os.path.basename(cand_dir)
        for wav in sorted(glob.glob(os.path.join(cand_dir, "*.wav"))):
            idx = int(os.path.splitext(os.path.basename(wav))[0])
            by_sentence[idx].append((label, wav))
    return by_sentence


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--sentence", type=int, help="이 문장만")
    ap.add_argument("--rater", default=os.environ.get("USER", "anon"))
    args = ap.parse_args()

    player = find_player()
    if not player:
        print("재생기를 못 찾았다 (afplay/aplay/ffplay 중 하나 필요)")
        return 1

    p = repo_paths()
    sentences = load_sentences(p["sentences"])
    by_sentence = collect_wavs(p["out"])
    if not by_sentence:
        print("out/tts/ 가 비었다. python3 run.py --tts 를 먼저 돌릴 것")
        return 1

    print(f"심사자: {args.rater}")
    print("점수 1~5 (1=못 듣겠다, 3=알아는 듣겠다, 5=사람 같다)")
    print("r=다시 듣기, s=건너뛰기, q=중단\n")

    scores = defaultdict(list)
    for idx in sorted(by_sentence):
        if args.sentence and idx != args.sentence:
            continue
        entries = by_sentence[idx][:]
        random.shuffle(entries)  # ← 편향 방지의 핵심

        print(f"\n[{idx}] {sentences[idx-1]}")
        for n, (label, wav) in enumerate(entries, 1):
            while True:
                print(f"    후보 {n}/{len(entries)} 재생…")
                subprocess.run(player + [wav], capture_output=True)
                ans = input("    점수(1-5) / r / s / q > ").strip().lower()
                if ans == "r":
                    continue
                if ans == "s":
                    break
                if ans == "q":
                    print("\n중단")
                    return 0
                if ans in "12345" and ans:
                    scores[label].append(int(ans))
                    break
                print("    1~5 / r / s / q 중 하나")

    print("\n" + "=" * 56)
    print("정답 공개")
    print("=" * 56)
    for label, vals in sorted(scores.items(), key=lambda kv: -sum(kv[1]) / len(kv[1])):
        print(f"    {label:<20} 평균 {sum(vals)/len(vals):.2f}  (n={len(vals)}) {vals}")

    path = os.path.join(p["out"], "blind_scores.json")
    prev = json.load(open(path, encoding="utf-8")) if os.path.exists(path) else []
    prev.append({"rater": args.rater, "scores": dict(scores)})
    with open(path, "w", encoding="utf-8") as f:
        json.dump(prev, f, ensure_ascii=False, indent=2)
    print(f"\n저장: {path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
