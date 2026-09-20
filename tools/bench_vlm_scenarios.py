#!/usr/bin/env python3
"""Benchmark project-specific Talking Lamp VLM scenarios and GPU memory.

The model and processor are loaded once, then every case is run sequentially with
one image tile. The raw answer, basic semantic checks, latency, and process-local
CUDA allocated/reserved peaks are retained in one JSON artifact.
"""
import argparse
import hashlib
import importlib.metadata
import json
import platform
import re
import time
import sys
from pathlib import Path


sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from cognition.contract import ALLOWED_MOTIONS, SYSTEM_PROMPT, CognitionResult


HAN_OR_KANA = re.compile(r"[\u3400-\u4dbf\u4e00-\u9fff\u3040-\u30ff]")
HANGUL = re.compile(r"[\uac00-\ud7af]")


def extract_object(text):
    """Return a JSON object even when the model surrounds it with prose/fences."""
    decoder = json.JSONDecoder()
    for match in re.finditer(r"\{", text):
        try:
            value, _ = decoder.raw_decode(text[match.start():])
        except json.JSONDecodeError:
            continue
        if isinstance(value, dict):
            return value
    return None


def evaluate(text, case):
    parsed = extract_object(text)
    observation = str((parsed or {}).get("observation", "")).strip()
    speech = str((parsed or {}).get("speech_ko", "")).strip()
    motion = str((parsed or {}).get("motion", "")).strip()
    folded = observation.casefold()
    groups = case.get("required_any", [])
    checks = {
        "json_object": parsed is not None,
        "exact_fields": parsed is not None and set(parsed) == {
            "observation", "speech_ko", "motion"
        },
        "visual_concepts": all(
            any(term.casefold() in folded for term in group) for group in groups
        ),
        "korean_speech": bool(HANGUL.search(speech)) and not HAN_OR_KANA.search(speech),
        "allowed_motion": motion in ALLOWED_MOTIONS,
        "forbidden_motion_absent": motion not in set(case.get("forbidden_motions", [])),
    }
    expected = case.get("expected_motions")
    if expected:
        checks["scenario_motion"] = motion in set(expected)
    try:
        handoff = CognitionResult.from_text(text).handoff()
    except (ValueError, TypeError):
        handoff = None
    checks["runtime_contract"] = handoff is not None
    return {
        "handoff": handoff,
        "parsed": parsed,
        "checks": checks,
        "passed": all(checks.values()),
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--model-id", default="OpenGVLab/InternVL3_5-1B-HF")
    parser.add_argument("--revision", default="main")
    parser.add_argument("--max-new-tokens", type=int, default=96)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    if args.max_new_tokens < 1:
        parser.error("max-new-tokens must be positive")

    manifest_path = args.manifest.resolve()
    repo_root = manifest_path.parent.parent
    cases = json.loads(manifest_path.read_text())["cases"]
    if not cases:
        parser.error("manifest must contain at least one case")

    import torch
    from PIL import Image, ImageOps
    from transformers import (
        AutoModelForImageTextToText,
        AutoProcessor,
        BitsAndBytesConfig,
    )
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA GPU required")
    torch.manual_seed(0)
    baseline_free, baseline_total = torch.cuda.mem_get_info()
    gpu_baseline_used = baseline_total - baseline_free
    quant = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_use_double_quant=True,
        bnb_4bit_compute_dtype=torch.float16,
    )
    processor = AutoProcessor.from_pretrained(args.model_id, revision=args.revision)
    processor.image_processor.crop_to_patches = False
    processor.image_processor.min_patches = 1
    processor.image_processor.max_patches = 1
    load_start = time.perf_counter()
    model = AutoModelForImageTextToText.from_pretrained(
        args.model_id,
        revision=args.revision,
        quantization_config=quant,
        device_map="cuda",
        low_cpu_mem_usage=True,
    ).eval()
    torch.cuda.synchronize()
    load_s = time.perf_counter() - load_start
    load_allocated = torch.cuda.memory_allocated()
    load_reserved = torch.cuda.memory_reserved()
    load_device_used = torch.cuda.mem_get_info()[1] - torch.cuda.mem_get_info()[0]

    system_prompt = SYSTEM_PROMPT
    results = []
    for index, case in enumerate(cases):
        image_path = (repo_root / case["image"]).resolve()
        with Image.open(image_path) as source:
            image = ImageOps.exif_transpose(source).convert("RGB")
        messages = [
            {"role": "system", "content": [{"type": "text", "text": system_prompt}]},
            {"role": "user", "content": [
                {"type": "image", "image": image},
                {"type": "text", "text": case["question"]},
            ]},
        ]
        inputs = processor.apply_chat_template(
            messages,
            add_generation_prompt=True,
            tokenize=True,
            return_dict=True,
            return_tensors="pt",
        ).to(model.device)
        tile_count = int(inputs["pixel_values"].shape[0])
        if tile_count != 1:
            raise RuntimeError(f"{case['id']}: expected one tile, got {tile_count}")
        visual_tokens = int((inputs["input_ids"] == model.config.image_token_id).sum())
        torch.cuda.synchronize()
        torch.cuda.reset_peak_memory_stats()
        started = time.perf_counter()
        with torch.inference_mode():
            output = model.generate(
                **inputs,
                max_new_tokens=args.max_new_tokens,
                do_sample=False,
            )
        torch.cuda.synchronize()
        infer_s = time.perf_counter() - started
        generated = output[0, inputs["input_ids"].shape[1]:]
        text = processor.decode(generated, skip_special_tokens=True).strip()
        judged = evaluate(text, case)
        record = {
            "index": index,
            "id": case["id"],
            "category": case["category"],
            "image": case["image"],
            "image_sha256": hashlib.sha256(image_path.read_bytes()).hexdigest(),
            "question": case["question"],
            "expected": {
                "required_any": case.get("required_any", []),
                "expected_motions": case.get("expected_motions", []),
                "forbidden_motions": case.get("forbidden_motions", []),
            },
            "tile_count": tile_count,
            "visual_tokens": visual_tokens,
            "input_tokens": int(inputs["input_ids"].shape[1]),
            "generated_tokens": int(generated.numel()),
            "infer_s": infer_s,
            "text": text,
            "cuda_peak_allocated_bytes": torch.cuda.max_memory_allocated(),
            "cuda_peak_reserved_bytes": torch.cuda.max_memory_reserved(),
            "device_used_after_inference_bytes": (
                torch.cuda.mem_get_info()[1] - torch.cuda.mem_get_info()[0]
            ),
            **judged,
        }
        results.append(record)
        print(json.dumps({
            "id": record["id"], "passed": record["passed"],
            "infer_s": record["infer_s"], "text": record["text"],
        }, ensure_ascii=False), flush=True)
        del image, messages, inputs, output, generated

    passed = sum(record["passed"] for record in results)
    result = {
        "schema_version": 1,
        "hardware_scope": "PC discrete GPU; not Jetson unified-memory measurement",
        "model_id": args.model_id,
        "resolved_revision": getattr(model.config, "_commit_hash", None),
        "gpu": torch.cuda.get_device_name(),
        "machine": platform.machine(),
        "versions": {name: importlib.metadata.version(name) for name in [
            "torch", "transformers", "bitsandbytes", "accelerate", "pillow"
        ]},
        "quantization": "bitsandbytes NF4 double-quant, fp16 compute",
        "manifest_sha256": hashlib.sha256(manifest_path.read_bytes()).hexdigest(),
        "system_prompt": system_prompt,
        "one_tile": True,
        "max_new_tokens": args.max_new_tokens,
        "load_s": load_s,
        "gpu_baseline_used_bytes": gpu_baseline_used,
        "load_cuda_allocated_bytes": load_allocated,
        "load_cuda_reserved_bytes": load_reserved,
        "device_used_after_load_bytes": load_device_used,
        "summary": {
            "case_count": len(results),
            "passed": passed,
            "failed": len(results) - passed,
            "pass_rate": passed / len(results),
            "mean_infer_s": sum(r["infer_s"] for r in results) / len(results),
            "max_infer_s": max(r["infer_s"] for r in results),
            "max_cuda_peak_allocated_bytes": max(r["cuda_peak_allocated_bytes"] for r in results),
            "max_cuda_peak_reserved_bytes": max(r["cuda_peak_reserved_bytes"] for r in results),
            "max_device_used_after_inference_bytes": max(
                r["device_used_after_inference_bytes"] for r in results
            ),
        },
        "cases": results,
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n")


if __name__ == "__main__":
    main()
