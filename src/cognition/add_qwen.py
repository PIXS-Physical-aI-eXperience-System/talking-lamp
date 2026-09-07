#!/usr/bin/env python3
"""bench_vlm.py 의 MODEL_REGISTRY 에 Qwen2-VL-2B-Instruct 추가.

vlm_bench/vlm_bench/ 에서 한 번 실행:
    python3 add_qwen.py

Qwen2-VL-2B-Instruct: Apache 2.0(상용 OK), 2B, 한국어 지원.
(제외했던 Qwen2.5-VL-3B = Qwen Research License(비상용)와는 다른 모델)

InternVL/SmolVLM/Gemma 와 동일한 AutoModelForImageTextToText 경로라 코드 수정 없이
--model qwen 으로 같은 조건 비교 가능. 원본은 bench_vlm.py.bak2 로 백업.
"""
import pathlib
import py_compile
import shutil
import sys

TARGET = pathlib.Path("bench_vlm.py")

OLD = '''    "smolvlm": {
        "id": "HuggingFaceTB/SmolVLM2-2.2B-Instruct",
        "license": "Apache 2.0",
        "note": "경량·최고속 기준점. 한국어는 약할 수 있음.",
    },'''

NEW = OLD + '''
    "qwen": {
        "id": "Qwen/Qwen2-VL-2B-Instruct",
        "license": "Apache 2.0",
        "note": "Qwen2-VL 2B, Apache 2.0(상용 OK), 한국어 지원. 제외한 Qwen2.5-VL-3B(비상용)와 다른 모델.",
    },'''


def main():
    if not TARGET.exists():
        sys.exit(f"[중단] {TARGET} 가 없습니다. vlm_bench/vlm_bench/ 안에서 실행하세요.")
    src = TARGET.read_text(encoding="utf-8")

    if '"qwen"' in src and "Qwen2-VL-2B-Instruct" in src:
        print("[건너뜀] 이미 qwen 이 등록되어 있습니다.")
        return
    if OLD not in src:
        print("[중단] smolvlm 등록 블록을 찾지 못했습니다. bench_vlm.py 가 예상과 다릅니다.")
        print("       파일을 수정하지 않았습니다. 이 메시지를 그대로 전달해주세요.")
        sys.exit(1)

    shutil.copy2(TARGET, TARGET.with_suffix(".py.bak2"))
    src = src.replace(OLD, NEW, 1)
    TARGET.write_text(src, encoding="utf-8")

    try:
        py_compile.compile(str(TARGET), doraise=True)
    except py_compile.PyCompileError as e:
        print("[경고] 문법 오류. 백업(bench_vlm.py.bak2)으로 되돌리세요.")
        print(e)
        sys.exit(1)

    print("[완료] qwen (Qwen/Qwen2-VL-2B-Instruct) 등록. 백업: bench_vlm.py.bak2")
    print("이제 --model qwen 사용 가능:")
    print("  python3 bench_vlm.py --model qwen --images-dir ./test_images --limit 1 --skip-quality-prompt")


if __name__ == "__main__":
    main()
