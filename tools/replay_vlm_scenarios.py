#!/usr/bin/env python3
"""Replay validated VLM benchmark answers on NullBackend and export TTS inputs."""
import argparse
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from cognition.contract import CognitionResult


def replay(document, out_dir):
    from motion.runtime import MotionRuntime, NullBackend

    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    records = []
    for index, case in enumerate(document["cases"]):
        record = {"id": case["id"], "passed": False}
        try:
            if case.get("passed") is not True:
                raise ValueError("benchmark case did not pass")
            result = CognitionResult.from_text(case["text"])
            runtime = MotionRuntime(backend=NullBackend())
            if result.motion != "idle":
                runtime.play_primitive(result.motion)
            for _ in range(100):
                state = runtime.step()
            tts_path = out_dir / f"{index:03d}-tts.json"
            tts_path.write_text(json.dumps(result.handoff()["tts"], ensure_ascii=False) + "\n")
            record.update(passed=True, handoff=result.handoff(),
                          tts_input=str(tts_path), q_cmd=state.q_cmd.tolist())
        except (ValueError, KeyError, TypeError) as exc:
            record["error"] = str(exc)
        records.append(record)
    return {"backend": "NullBackend", "hardware_executed": False, "cases": records,
            "passed": bool(records) and all(r["passed"] for r in records)}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    args = parser.parse_args()
    report = replay(json.loads(args.input.read_text()), args.out_dir)
    (args.out_dir / "replay.json").write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps(report, ensure_ascii=False))
    return 0 if report["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
