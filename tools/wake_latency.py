#!/usr/bin/env python3
"""Full cold-start latency from a pre-quantized checkpoint: load + first inference,
one-tile + short output, simulating a wake-word -> response cycle."""
import argparse
import hashlib
import json
import time
from pathlib import Path


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--model-dir', type=Path, required=True)
    ap.add_argument('--image', type=Path, required=True)
    ap.add_argument('--max-new-tokens', type=int, default=32)
    ap.add_argument('--prompt', default='이 사진에 뭐가 보여?')
    ap.add_argument('--out', type=Path)
    args = ap.parse_args()

    t_start = time.perf_counter()
    import torch
    from PIL import Image, ImageOps
    from transformers import AutoProcessor, AutoModelForImageTextToText

    processor = AutoProcessor.from_pretrained(args.model_dir)
    processor.image_processor.crop_to_patches = False
    processor.image_processor.min_patches = 1
    processor.image_processor.max_patches = 1

    t0 = time.perf_counter()
    model = AutoModelForImageTextToText.from_pretrained(
        args.model_dir, device_map='cuda', low_cpu_mem_usage=True).eval()
    torch.cuda.synchronize()
    load_s = time.perf_counter() - t0

    with Image.open(args.image) as source:
        img = ImageOps.exif_transpose(source).convert('RGB')
    messages = [{'role': 'user', 'content': [
        {'type': 'image', 'image': img}, {'type': 'text', 'text': args.prompt}]}]
    inputs = processor.apply_chat_template(messages, add_generation_prompt=True,
                tokenize=True, return_dict=True, return_tensors='pt').to(model.device)
    torch.cuda.synchronize()
    t1 = time.perf_counter()
    with torch.inference_mode():
        output = model.generate(**inputs, max_new_tokens=args.max_new_tokens, do_sample=False)
    torch.cuda.synchronize()
    infer_s = time.perf_counter() - t1
    generated = output[0, inputs['input_ids'].shape[1]:]
    text = processor.decode(generated, skip_special_tokens=True)
    total_s = time.perf_counter() - t_start

    result = {
        'schema_version': 1,
        'model_dir': str(args.model_dir),
        'image_sha256': hashlib.sha256(args.image.read_bytes()).hexdigest(),
        'prompt': args.prompt,
        'one_tile': True,
        'max_new_tokens': args.max_new_tokens,
        'load_s': load_s,
        'total_wake_to_response_s': total_s,
        'runs': [{
            'kind': 'first_inference',
            'infer_s': infer_s,
            'generated_tokens': int(generated.numel()),
            'text': text,
            'cuda_peak_allocated_bytes': torch.cuda.max_memory_allocated(),
            'cuda_peak_reserved_bytes': torch.cuda.max_memory_reserved(),
        }],
    }
    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(json.dumps(result, ensure_ascii=False, indent=2) + '\n')
    print(json.dumps(result, ensure_ascii=False), flush=True)


if __name__ == '__main__':
    main()
