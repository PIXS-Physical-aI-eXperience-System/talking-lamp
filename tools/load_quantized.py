#!/usr/bin/env python3
"""Time a cold load from an already-quantized checkpoint (no on-the-fly quant)."""
import argparse
import time
from pathlib import Path


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--model-dir', type=Path, required=True)
    args = ap.parse_args()

    import torch
    from transformers import AutoProcessor, AutoModelForImageTextToText

    processor = AutoProcessor.from_pretrained(args.model_dir)
    t0 = time.perf_counter()
    model = AutoModelForImageTextToText.from_pretrained(
        args.model_dir, device_map='cuda', low_cpu_mem_usage=True).eval()
    torch.cuda.synchronize()
    load_s = time.perf_counter() - t0
    print(f'load from pre-quantized checkpoint: {load_s:.1f}s')
    print(f'cuda_allocated_after_load_gb: {torch.cuda.memory_allocated()/1e9:.2f}')


if __name__ == '__main__':
    main()
