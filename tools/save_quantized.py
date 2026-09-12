#!/usr/bin/env python3
"""Load with bnb 4bit once, then save the quantized state to disk so future
loads can skip the on-the-fly fp16->nf4 quantization step."""
import argparse
import time
from pathlib import Path


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--model-id', default='OpenGVLab/InternVL3_5-2B-HF')
    ap.add_argument('--out', type=Path, required=True)
    args = ap.parse_args()

    import torch
    from transformers import AutoProcessor, AutoModelForImageTextToText, BitsAndBytesConfig

    quant = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type='nf4',
                                bnb_4bit_use_double_quant=True,
                                bnb_4bit_compute_dtype=torch.float16)
    processor = AutoProcessor.from_pretrained(args.model_id)
    t0 = time.perf_counter()
    model = AutoModelForImageTextToText.from_pretrained(
        args.model_id, quantization_config=quant, device_map='cuda',
        low_cpu_mem_usage=True).eval()
    torch.cuda.synchronize()
    quant_load_s = time.perf_counter() - t0
    print(f'initial bnb load+quantize: {quant_load_s:.1f}s')

    args.out.mkdir(parents=True, exist_ok=True)
    t0 = time.perf_counter()
    model.save_pretrained(args.out, safe_serialization=True)
    processor.save_pretrained(args.out)
    save_s = time.perf_counter() - t0
    print(f'save_pretrained: {save_s:.1f}s -> {args.out}')


if __name__ == '__main__':
    main()
