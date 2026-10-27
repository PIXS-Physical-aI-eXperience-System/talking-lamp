#!/usr/bin/env python3
"""VLM 메모리 오버헤드 분해 + 편차 측정 (Jetson venv에서 실행).

총 메모리가 "모델 가중치 / CUDA·프레임워크 오버헤드 / 추론 활성화·KV" 중 각각
얼마인지 쪼개고, 추론을 여러 번 반복해 편차까지 본다.

두 가지 메모리 지표를 같이 낸다:
  - /proc/meminfo(MemAvailable) 기반 : 시스템 전체 관점(프레임워크·CUDA 컨텍스트 포함).
      단, 로딩 직후 페이지캐시/양자화 임시버퍼에 흔들려 편차가 큼(특히 '가중치').
  - torch.cuda.max_memory_allocated 기반 : PyTorch가 실제 할당한 텐서 메모리만.
      페이지캐시·다른 프로세스에 안 흔들려 **신뢰도 높음**. (CUDA 컨텍스트 고정비용은 미포함)

사용:
  python3 mem_breakdown.py --model-id Qwen/Qwen2-VL-2B-Instruct --images-dir ./test_images --img-max-side 448 --repeat 5 --out results/qwen_448.json
  python3 mem_breakdown.py --model-id OpenGVLab/InternVL3_5-2B-HF --images-dir ./test_images --img-max-side 448 --repeat 5 --out results/internvl_448.json
"""
from __future__ import annotations

import argparse
import json
import threading
import time
from pathlib import Path


def mem_avail_kb():
    try:
        with open("/proc/meminfo") as f:
            for line in f:
                if line.startswith("MemAvailable:"):
                    return int(line.split()[1])
    except FileNotFoundError:
        return None
    return None


class PeakSampler:
    """구간 동안 MemAvailable 최소값(=메모리 최대 점유 시점)을 추적."""
    def __init__(self, interval=0.05):
        self.interval = interval
        self._stop = threading.Event()
        self._t = None
        self.min_avail = None

    def _run(self):
        while not self._stop.is_set():
            a = mem_avail_kb()
            if a is not None and (self.min_avail is None or a < self.min_avail):
                self.min_avail = a
            time.sleep(self.interval)

    def start(self):
        self.min_avail = mem_avail_kb()
        self._stop.clear()
        self._t = threading.Thread(target=self._run, daemon=True)
        self._t.start()

    def stop(self):
        self._stop.set()
        if self._t:
            self._t.join(timeout=1.0)
        return self.min_avail


def gb(kb):
    return round(kb / (1024.0 * 1024.0), 3)


def mb2gb(mb):
    return round(mb / 1024.0, 3)


def pick_image(images_dir, image):
    if image:
        return Path(image)
    d = Path(images_dir)
    exts = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}
    imgs = sorted(p for p in d.iterdir() if p.suffix.lower() in exts)
    if not imgs:
        raise SystemExit(f"[오류] {images_dir} 에 이미지가 없습니다.")
    return imgs[0]


def resize_max_side(img, max_side):
    if not max_side:
        return img
    w, h = img.size
    m = max(w, h)
    if m <= max_side:
        return img
    s = max_side / float(m)
    return img.resize((max(1, int(w * s)), max(1, int(h * s))))


def summarize(vals):
    return {"min": round(min(vals), 3), "avg": round(sum(vals) / len(vals), 3), "max": round(max(vals), 3)}


def main():
    ap = argparse.ArgumentParser(description="VLM 메모리 오버헤드 분해 + 편차")
    ap.add_argument("--model-id", default="Qwen/Qwen2-VL-2B-Instruct")
    ap.add_argument("--images-dir", default="./test_images")
    ap.add_argument("--image", default=None)
    ap.add_argument("--img-max-side", type=int, default=None, help="이미지 축소(경량화) 후 측정")
    ap.add_argument("--max-new-tokens", type=int, default=128)
    ap.add_argument("--repeat", type=int, default=1, help="한 번 로딩 후 추론 반복 횟수(편차 측정)")
    ap.add_argument("--out", default=None, help="결과 JSON 저장 경로")
    args = ap.parse_args()

    img_path = pick_image(args.images_dir, args.image)

    base = mem_avail_kb()
    if base is None:
        raise SystemExit("[오류] /proc/meminfo 를 읽을 수 없습니다 (Linux/Jetson에서 실행하세요).")

    import torch
    from PIL import Image
    from transformers import AutoModelForImageTextToText, AutoProcessor, BitsAndBytesConfig

    orig_size = Image.open(img_path).size  # (w, h) 원본

    # (1) CUDA/프레임워크 초기화
    torch.zeros(1).cuda()
    torch.cuda.synchronize()
    after_cuda = mem_avail_kb()

    # (2) 모델 로딩 (INT4)
    print(f"[로딩] {args.model_id} (INT4) ...")
    quant = BitsAndBytesConfig(
        load_in_4bit=True, bnb_4bit_quant_type="nf4",
        bnb_4bit_use_double_quant=True, bnb_4bit_compute_dtype=torch.float16,
    )
    torch.cuda.reset_peak_memory_stats()
    t0 = time.perf_counter()
    processor = AutoProcessor.from_pretrained(args.model_id, trust_remote_code=True)
    model = AutoModelForImageTextToText.from_pretrained(
        args.model_id, quantization_config=quant, device_map="cuda",
        trust_remote_code=True, low_cpu_mem_usage=True,
    ).eval()
    torch.cuda.synchronize()
    load_s = round(time.perf_counter() - t0, 1)
    after_load = mem_avail_kb()
    cuda_weights_mb = torch.cuda.memory_allocated() / (1024 * 1024)  # 페이지캐시 무관 정확값

    # 입력 준비 (반복 동안 동일)
    img = resize_max_side(Image.open(img_path).convert("RGB"), args.img_max_side)
    proc_size = img.size
    messages = [{"role": "user", "content": [
        {"type": "image", "image": img},
        {"type": "text", "text": "이 사진을 한국어로 설명해줘"},
    ]}]
    inputs = processor.apply_chat_template(
        messages, add_generation_prompt=True, tokenize=True,
        return_dict=True, return_tensors="pt",
    ).to(model.device)

    # (3) 추론 반복
    runs = []
    for i in range(args.repeat):
        torch.cuda.reset_peak_memory_stats()
        sampler = PeakSampler()
        sampler.start()
        t1 = time.perf_counter()
        with torch.no_grad():
            model.generate(**inputs, max_new_tokens=args.max_new_tokens, do_sample=False)
        torch.cuda.synchronize()
        infer_s = time.perf_counter() - t1
        min_avail = sampler.stop()
        cuda_peak_mb = torch.cuda.max_memory_allocated() / (1024 * 1024)
        sys_total_gb = gb(base - min_avail)  # 시스템 관점 총 피크(프레임워크 포함)
        runs.append({
            "infer_s": round(infer_s, 2),
            "cuda_peak_gb": mb2gb(cuda_peak_mb),
            "sys_total_gb": sys_total_gb,
        })
        print(f"  [추론 {i+1}/{args.repeat}] {infer_s:.2f}s | "
              f"cuda_peak {mb2gb(cuda_peak_mb)}GB | sys_total {sys_total_gb}GB")

    # 시스템 관점 분해 (1회차 로딩 기준; 편차 있음)
    fw_gb = gb(base - after_cuda)
    weight_sys_gb = gb(after_cuda - after_load)

    result = {
        "model_id": args.model_id,
        "orig_image_size": list(orig_size),
        "proc_image_size": list(proc_size),
        "img_max_side": args.img_max_side,
        "max_new_tokens": args.max_new_tokens,
        "repeat": args.repeat,
        "load_s": load_s,
        # 신뢰값 (torch.cuda, 페이지캐시 무관)
        "cuda_weights_gb": mb2gb(cuda_weights_mb),
        "cuda_peak_gb": summarize([r["cuda_peak_gb"] for r in runs]),
        # 시스템 관점 (프레임워크 포함, 편차 큼)
        "sys_framework_gb": fw_gb,
        "sys_weights_gb": weight_sys_gb,
        "sys_total_gb": summarize([r["sys_total_gb"] for r in runs]),
        "infer_s": summarize([r["infer_s"] for r in runs]),
        "runs": runs,
    }

    print("\n========== 메모리 분해 / 편차 ==========")
    print(f"모델        : {args.model_id}")
    print(f"이미지      : {img_path.name}  원본 {orig_size[0]}x{orig_size[1]} → 처리 {proc_size[0]}x{proc_size[1]}")
    print(f"로딩 시간   : {load_s}s   추론 {args.repeat}회")
    print("--- 신뢰값 (torch.cuda, 페이지캐시 무관) ---")
    print(f"  모델 가중치(텐서)   : {result['cuda_weights_gb']} GB")
    print(f"  추론 피크(텐서)     : min {result['cuda_peak_gb']['min']} / avg {result['cuda_peak_gb']['avg']} / max {result['cuda_peak_gb']['max']} GB")
    print("--- 시스템 관점 (프레임워크 포함, 편차 큼) ---")
    print(f"  CUDA/프레임워크     : {fw_gb} GB")
    print(f"  총 피크(시스템)     : min {result['sys_total_gb']['min']} / avg {result['sys_total_gb']['avg']} / max {result['sys_total_gb']['max']} GB")
    print(f"  추론시간            : min {result['infer_s']['min']} / avg {result['infer_s']['avg']} / max {result['infer_s']['max']} s")
    print("========================================")

    if args.out:
        outp = Path(args.out)
        outp.parent.mkdir(parents=True, exist_ok=True)
        outp.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
        print(f"[저장] {outp}")


if __name__ == "__main__":
    main()
