#!/usr/bin/env python3
"""Translate one saved English VLM answer to Korean with NLLB."""
import argparse
import json
import time
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--model-dir', type=Path, required=True)
    parser.add_argument('--input', type=Path, required=True)
    parser.add_argument('--out', type=Path, required=True)
    parser.add_argument('--max-new-tokens', type=int, default=64)
    args = parser.parse_args()

    import torch
    from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

    source = json.loads(args.input.read_text())
    source_text = source['runs'][0]['text']
    started = time.perf_counter()
    tokenizer = AutoTokenizer.from_pretrained(args.model_dir, src_lang='eng_Latn')
    model = AutoModelForSeq2SeqLM.from_pretrained(
        args.model_dir, dtype=torch.float16, low_cpu_mem_usage=True).to('cuda').eval()
    torch.cuda.synchronize()
    load_s = time.perf_counter() - started

    inputs = tokenizer(source_text, return_tensors='pt').to('cuda')
    started = time.perf_counter()
    with torch.inference_mode():
        output = model.generate(
            **inputs,
            forced_bos_token_id=tokenizer.convert_tokens_to_ids('kor_Hang'),
            max_new_tokens=args.max_new_tokens,
            do_sample=False)
    torch.cuda.synchronize()
    infer_s = time.perf_counter() - started
    translated = tokenizer.decode(output[0], skip_special_tokens=True)
    result = {
        'schema_version': 1,
        'source': str(args.input),
        'source_text': source_text,
        'language': 'ko',
        'text': translated,
        'load_s': load_s,
        'infer_s': infer_s,
        'generated_tokens': int(output[0].numel()),
        'cuda_peak_allocated_bytes': torch.cuda.max_memory_allocated(),
        'cuda_peak_reserved_bytes': torch.cuda.max_memory_reserved(),
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(result, ensure_ascii=False, indent=2) + '\n')
    print(json.dumps(result, ensure_ascii=False), flush=True)


if __name__ == '__main__':
    main()
