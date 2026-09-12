#!/usr/bin/env python3
"""LMDeploy TurboMind memory/latency probe, comparable to bench_vlm_ram.py.

TurboMind manages its own CUDA allocator and pre-allocates a KV cache pool, so
its GPU footprint is not read from torch. GPU bytes here come from nvidia-smi
total used memory on an otherwise idle GPU; report cache_max_entry_count with it.
Host RAM and GPU bytes are separate. Not a Jetson measurement.
"""
import argparse
import gc
import hashlib
import importlib.metadata
import json
import platform
import subprocess
import threading
import time
from pathlib import Path


def gpu_used_bytes():
    out = subprocess.run(
        ['nvidia-smi', '--query-gpu=memory.used', '--format=csv,noheader,nounits'],
        capture_output=True, text=True, check=True).stdout.strip().splitlines()
    return int(out[0]) * 1024 * 1024


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--model-id', default='OpenGVLab/InternVL3_5-2B-HF')
    ap.add_argument('--image', type=Path, required=True)
    ap.add_argument('--prompt', default='이 사진을 한국어로 설명해줘')
    ap.add_argument('--max-new-tokens', type=int, default=128)
    ap.add_argument('--cache-max-entry-count', type=float, default=0.1)
    ap.add_argument('--session-len', type=int, default=8192)
    ap.add_argument('--repeat', type=int, default=5)
    ap.add_argument('--out', type=Path, required=True)
    args = ap.parse_args()

    import psutil
    process = psutil.Process()
    stop = threading.Event()
    samples = []
    phase = 'imports'

    def sample():
        while not stop.is_set():
            try:
                g = gpu_used_bytes()
            except Exception:
                g = -1
            samples.append({'phase': phase, 'rss_bytes': process.memory_info().rss,
                            'gpu_used_bytes': g, 'time_s': time.monotonic()})
            stop.wait(.05)

    sampler = threading.Thread(target=sample, daemon=True)
    sampler.start()
    try:
        gpu_idle = gpu_used_bytes()
        from lmdeploy import pipeline, TurbomindEngineConfig, GenerationConfig
        from lmdeploy.vl import load_image
        img = load_image(str(args.image))
        phase = 'load'
        t0 = time.perf_counter()
        pipe = pipeline(args.model_id, backend_config=TurbomindEngineConfig(
            cache_max_entry_count=args.cache_max_entry_count,
            session_len=args.session_len))
        load_s = time.perf_counter() - t0
        gpu_after_load = gpu_used_bytes()
        gen = GenerationConfig(max_new_tokens=args.max_new_tokens, do_sample=False)
        runs = []
        for i in range(args.repeat + 1):
            phase = 'first_inference' if i == 0 else 'warm_inference'
            start = time.perf_counter()
            resp = pipe((args.prompt, img), gen_config=gen)
            elapsed = time.perf_counter() - start
            runs.append({'kind': phase, 'infer_s': elapsed,
                         'generated_tokens': resp.generate_token_len,
                         'input_tokens': resp.input_token_len,
                         'text': resp.text,
                         'gpu_used_bytes': gpu_used_bytes()})
            print(json.dumps(runs[-1], ensure_ascii=False), flush=True)
        gpu_peak = max(s['gpu_used_bytes'] for s in samples if s['gpu_used_bytes'] > 0)
        phase = 'unload'
        del pipe
        gc.collect()
        time.sleep(.5)
        result = {
            'schema_version': 1, 'hardware_scope': 'PC dGPU; not Jetson measurement',
            'engine': 'lmdeploy-turbomind', 'model_id': args.model_id,
            'gpu': subprocess.run(['nvidia-smi', '--query-gpu=name', '--format=csv,noheader'],
                                  capture_output=True, text=True).stdout.strip(),
            'machine': platform.machine(),
            'versions': {k: importlib.metadata.version(k) for k in
                         ['lmdeploy', 'torch', 'transformers', 'torchvision']},
            'image_sha256': hashlib.sha256(args.image.read_bytes()).hexdigest(),
            'prompt': args.prompt, 'max_new_tokens': args.max_new_tokens,
            'cache_max_entry_count': args.cache_max_entry_count,
            'session_len': args.session_len,
            'gpu_idle_bytes': gpu_idle, 'gpu_after_load_bytes': gpu_after_load,
            'gpu_peak_used_bytes': gpu_peak, 'load_s': load_s, 'runs': runs,
            'after_unload_gpu_used_bytes': gpu_used_bytes(),
            'after_unload_rss_bytes': process.memory_info().rss}
    finally:
        stop.set()
        sampler.join()
    result['rss_peak_bytes_by_phase'] = {
        p: max(s['rss_bytes'] for s in samples if s['phase'] == p)
        for p in sorted({s['phase'] for s in samples})}
    result['samples'] = samples
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(result, ensure_ascii=False, indent=2) + '\n')


if __name__ == '__main__':
    main()
