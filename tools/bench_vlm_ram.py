#!/usr/bin/env python3
"""GPU comparison derived from docs/vlm-benchmark:src/cognition/mem_breakdown.py.

Run each profile in a separate process. Host RAM and GPU bytes are separate.
Download models before running with HF_HUB_OFFLINE=1.
"""
import argparse
import gc
import hashlib
import importlib.metadata
import json
import platform
import threading
import time
from pathlib import Path


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--model-id', default='OpenGVLab/InternVL3_5-2B-HF')
    ap.add_argument('--revision', default='main')
    ap.add_argument('--image', type=Path, required=True)
    ap.add_argument('--one-tile', action='store_true')
    ap.add_argument('--max-new-tokens', type=int, default=128)
    ap.add_argument('--prompt', default='이 사진을 한국어로 설명해줘')
    ap.add_argument('--repeat', type=int, default=5)
    ap.add_argument('--out', type=Path, required=True)
    args = ap.parse_args()
    if args.repeat < 1 or args.max_new_tokens < 1:
        ap.error('repeat and max-new-tokens must be positive')
    import psutil
    process = psutil.Process()
    stop = threading.Event()
    samples = []
    phase = 'imports'

    def sample():
        while not stop.is_set():
            samples.append({'phase': phase, 'rss_bytes': process.memory_info().rss,
                            'time_s': time.monotonic()})
            stop.wait(.02)

    sampler = threading.Thread(target=sample, daemon=True)
    sampler.start()
    try:
        import torch
        from PIL import Image, ImageOps
        from transformers import AutoProcessor, AutoModelForImageTextToText, BitsAndBytesConfig
        if not torch.cuda.is_available():
            raise RuntimeError('CUDA GPU required')
        torch.manual_seed(0)
        quant = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type='nf4',
                                  bnb_4bit_use_double_quant=True,
                                  bnb_4bit_compute_dtype=torch.float16)
        phase = 'load'
        t0 = time.perf_counter()
        processor = AutoProcessor.from_pretrained(args.model_id, revision=args.revision)
        if args.one_tile:
            processor.image_processor.crop_to_patches = False
            processor.image_processor.min_patches = 1
            processor.image_processor.max_patches = 1
        model = AutoModelForImageTextToText.from_pretrained(
            args.model_id, revision=args.revision, quantization_config=quant,
            device_map='cuda', low_cpu_mem_usage=True).eval()
        torch.cuda.synchronize()
        load_s = time.perf_counter() - t0
        load_peak = torch.cuda.max_memory_allocated()
        phase = 'prepare'
        with Image.open(args.image) as source:
            img = ImageOps.exif_transpose(source).convert('RGB')
        messages = [{'role': 'user', 'content': [
            {'type': 'image', 'image': img}, {'type': 'text', 'text': args.prompt}]}]
        inputs = processor.apply_chat_template(messages, add_generation_prompt=True,
                    tokenize=True, return_dict=True, return_tensors='pt').to(model.device)
        shapes = {k: list(v.shape) for k, v in inputs.items() if hasattr(v, 'shape')}
        tile_count = int(inputs['pixel_values'].shape[0])
        if args.one_tile and tile_count != 1:
            raise RuntimeError(f'Expected 1 tile, got {tile_count}')
        visual_tokens = int((inputs['input_ids'] == model.config.image_token_id).sum())
        resolved_revision = getattr(model.config, '_commit_hash', None)
        runs = []
        for i in range(args.repeat + 1):
            phase = 'first_inference' if i == 0 else 'warm_inference'
            torch.cuda.synchronize()
            torch.cuda.reset_peak_memory_stats()
            start = time.perf_counter()
            with torch.inference_mode():
                output = model.generate(**inputs, max_new_tokens=args.max_new_tokens, do_sample=False)
            torch.cuda.synchronize()
            elapsed = time.perf_counter() - start
            generated = output[0, inputs['input_ids'].shape[1]:]
            runs.append({'kind': phase, 'infer_s': elapsed,
                         'generated_tokens': generated.numel(),
                         'text': processor.decode(generated, skip_special_tokens=True),
                         'cuda_peak_allocated_bytes': torch.cuda.max_memory_allocated(),
                         'cuda_peak_reserved_bytes': torch.cuda.max_memory_reserved()})
            print(json.dumps(runs[-1], ensure_ascii=False), flush=True)
            del output, generated
        phase = 'unload'
        del model, processor, inputs, img
        gc.collect()
        torch.cuda.empty_cache()
        torch.cuda.synchronize()
        time.sleep(.1)
        result = {'schema_version': 1, 'hardware_scope': 'PC dGPU; not Jetson measurement',
                  'model_id': args.model_id, 'resolved_revision': resolved_revision,
                  'gpu': torch.cuda.get_device_name(), 'machine': platform.machine(),
                  'versions': {k: importlib.metadata.version(k) for k in
                               ['torch', 'transformers', 'bitsandbytes', 'accelerate', 'pillow']},
                  'image_sha256': hashlib.sha256(args.image.read_bytes()).hexdigest(),
                  'prompt': args.prompt, 'one_tile': args.one_tile,
                  'max_new_tokens': args.max_new_tokens, 'input_shapes': shapes,
                  'tile_count': tile_count, 'visual_tokens': visual_tokens,
                  'load_s': load_s, 'load_cuda_peak_allocated_bytes': load_peak,
                  'runs': runs, 'after_unload_cuda_allocated_bytes': torch.cuda.memory_allocated(),
                  'after_unload_cuda_reserved_bytes': torch.cuda.memory_reserved(),
                  'after_unload_rss_bytes': process.memory_info().rss}
    finally:
        stop.set()
        sampler.join()
    result['rss_peak_bytes_by_phase'] = {
        p: max(s['rss_bytes'] for s in samples if s['phase'] == p)
        for p in sorted({s['phase'] for s in samples})}
    result['rss_samples'] = samples
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(result, ensure_ascii=False, indent=2) + '\n')


if __name__ == '__main__':
    main()
