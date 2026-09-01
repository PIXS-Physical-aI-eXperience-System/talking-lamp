"""후보 비교 오케스트레이터.

각 후보를 '자기 venv의 파이썬으로 별도 프로세스에서' 돌린다. 이유가 두 가지다.

  1. 후보마다 의존성이 충돌한다 (MeloTTS는 특히 까다롭다). 한 venv에 다 넣으면 안 붙는다.
  2. 피크 RSS는 프로세스 단위로 재야 깨끗하다. 한 프로세스에서 여러 모델을 로드하면
     누가 얼마를 먹었는지 분리가 안 된다.

    python3 run.py --stt
    python3 run.py --tts
    python3 run.py --only fw-small-int8

표준 라이브러리만 쓴다 — 시스템 파이썬에서 그냥 돌아가야 한다.
"""
import argparse
import json
import os
import subprocess
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)   # voice-bench/ — 공용 모듈과 산출물이 있는 곳
sys.path.insert(0, ROOT)

from common import RESULT_MARKER, cer, load_sentences, repo_paths  # noqa: E402


def venv_python(venv_rel: str) -> str:
    return os.path.join(ROOT, venv_rel, "bin", "python")


def run_candidate(cand: dict, kind: str, ref_dir: str) -> dict:
    py = venv_python(cand["venv"])
    if not os.path.exists(py):
        return {"label": cand["label"], "error": f"venv 없음: {cand['venv']} — setup.sh 먼저 실행"}

    cmd = [py, os.path.join(ROOT, cand["runner"]), "--label", cand["label"]]
    # 상대 경로 인자를 voice-bench/ 기준으로 펴준다
    args = list(cand["args"])
    for i, a in enumerate(args):
        if i > 0 and args[i - 1] in ("--model-path", "--model", "--bin") and not os.path.isabs(a):
            cand_path = os.path.join(ROOT, a)
            if os.path.exists(cand_path):
                args[i] = cand_path
    cmd += args

    if kind == "stt":
        cmd += ["--ref-dir", ref_dir]
    else:
        cmd += ["--out-dir", os.path.join(ROOT, "out", "tts", cand["label"])]

    print(f"  ▶ {cand['label']} … ", end="", flush=True)
    proc = subprocess.run(cmd, capture_output=True, text=True, cwd=ROOT)

    for line in proc.stdout.splitlines():
        if line.startswith(RESULT_MARKER):
            print("완료")
            return json.loads(line[len(RESULT_MARKER):])

    print("실패")
    tail = "\n".join((proc.stderr or proc.stdout).strip().splitlines()[-6:])
    return {"label": cand["label"], "error": tail or f"exit {proc.returncode}, 출력 없음"}


def report_stt(results: list[dict], sentences: list[str]) -> None:
    print("\n" + "=" * 72)
    print("STT — 낮을수록 좋음(CER). RSS/TTFB는 이 맥 기준이라 Jetson에서 다시 재야 함")
    print("=" * 72)
    print(f"{'후보':<20} {'CER':>8} {'피크RSS':>10} {'평균TTFB':>10} {'평균추론':>10}")
    print("-" * 72)

    for r in results:
        if "error" in r:
            print(f"{r['label']:<20} {'—':>8} {'—':>10} {'—':>10} {'—':>10}   ! 실패")
            for line in r["error"].splitlines():
                print(f"        {line}")
            continue
        cers = [cer(ref, hyp) for ref, hyp in zip(sentences, r["transcripts"])]
        r["cer_per_sentence"] = [round(c, 4) for c in cers]
        r["cer_mean"] = round(sum(cers) / len(cers), 4)
        ttfb = [t for t in r["ttfb_s"] if t >= 0]
        print(f"{r['label']:<20} {r['cer_mean']*100:>7.1f}% "
              f"{r['peak_rss_mb']:>9.0f}M {sum(ttfb)/max(len(ttfb),1):>9.2f}s "
              f"{sum(r['infer_s'])/len(r['infer_s']):>9.2f}s")

    print("\n문장별 인식 결과")
    print("-" * 72)
    for i, ref in enumerate(sentences):
        print(f"\n[{i+1}] 정답: {ref}")
        for r in results:
            if "error" in r:
                continue
            mark = "✓" if r["cer_per_sentence"][i] == 0 else " "
            print(f"    {mark} {r['label']:<18} {r['cer_per_sentence'][i]*100:>5.1f}%  "
                  f"{r['transcripts'][i]}")

    print("\n" + "-" * 72)
    print("예산 대비: 음성 인식 할당 0.25 GB = 256 MB")
    for r in results:
        if "error" in r:
            continue
        over = r["peak_rss_mb"] / 256
        verdict = "예산 내" if over <= 1 else f"예산의 {over:.1f}배"
        print(f"    {r['label']:<20} {r['peak_rss_mb']:>7.0f} MB  → {verdict}")


def report_tts(results: list[dict], sentences: list[str]) -> None:
    print("\n" + "=" * 72)
    print("TTS — RTF < 1.0 이면 실시간보다 빠름. 음질은 blind.py 로 따로 심사")
    print("=" * 72)
    print(f"{'후보':<20} {'피크RSS':>10} {'평균RTF':>10} {'로드':>8}")
    print("-" * 72)
    for r in results:
        if "error" in r:
            print(f"{r['label']:<20} {'—':>10} {'—':>10} {'—':>8}   ! 실패")
            for line in r["error"].splitlines():
                print(f"        {line}")
            continue
        rtf = [s / a for s, a in zip(r["synth_s"], r["audio_s"]) if a > 0]
        r["rtf_mean"] = round(sum(rtf) / len(rtf), 3)
        print(f"{r['label']:<20} {r['peak_rss_mb']:>9.0f}M {r['rtf_mean']:>10.2f} "
              f"{r['load_s']:>7.1f}s")
        if r.get("rss_note"):
            print(f"    ※ {r['rss_note']}")

    print("\n예산 대비: 음성 합성 할당 0.2 GB = 205 MB")
    for r in results:
        if "error" in r:
            continue
        over = r["peak_rss_mb"] / 205
        print(f"    {r['label']:<20} {r['peak_rss_mb']:>7.0f} MB  → "
              f"{'예산 내' if over <= 1 else f'예산의 {over:.1f}배'}")
    if any("error" not in r for r in results):
        print("\n다음: python3 blind.py   (누가 만든 소린지 모르고 듣기)")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--stt", action="store_true")
    ap.add_argument("--tts", action="store_true")
    ap.add_argument("--only", help="이 라벨 하나만")
    args = ap.parse_args()
    if not args.stt and not args.tts:
        args.stt = args.tts = True

    p = repo_paths()
    sentences = load_sentences(p["sentences"])
    cfg = json.load(open(os.path.join(HERE, "candidates.json"), encoding="utf-8"))
    os.makedirs(p["out"], exist_ok=True)
    all_results = {}

    for kind, want in (("stt", args.stt), ("tts", args.tts)):
        if not want:
            continue
        cands = [c for c in cfg[kind] if c.get("enabled") and
                 (not args.only or c["label"] == args.only)]
        if not cands:
            continue

        if kind == "stt":
            n_ref = len([f for f in os.listdir(p["ref"]) if f.endswith(".wav")])
            if n_ref != len(sentences):
                print(f"! ref/ 에 wav가 {n_ref}개인데 문장은 {len(sentences)}개다.")
                print("  python3 record.py 로 6문장을 먼저 녹음할 것.")
                return 1

        print(f"\n[{kind.upper()}] 후보 {len(cands)}개")
        results = [run_candidate(c, kind, p["ref"]) for c in cands]
        (report_stt if kind == "stt" else report_tts)(results, sentences)
        all_results[kind] = results

    out_path = os.path.join(p["out"], "results.json")
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump({"sentences": sentences, "results": all_results}, f,
                  ensure_ascii=False, indent=2)
    print(f"\n원자료: {out_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
