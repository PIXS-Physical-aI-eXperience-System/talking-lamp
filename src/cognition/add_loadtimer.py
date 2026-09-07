#!/usr/bin/env python3
"""bench_vlm.py 의 build_model 에 로딩 시간(오버헤드) 측정 추가.

vlm_bench/vlm_bench/ 에서 한 번 실행:
    python3 add_loadtimer.py

모델 로딩/초기화에 걸린 시간을 콘솔에 `[로딩 완료] ... X.Xs` 로 찍는다.
(첫 실행은 HF 다운로드 시간까지 포함되므로 더 길다.)
원본은 bench_vlm.py.bak3 로 백업.
"""
import pathlib
import py_compile
import shutil
import sys

TARGET = pathlib.Path("bench_vlm.py")

PATCHES = [
    ("로딩 타이머 시작",
     '    print(f"[로딩] {cfg[\'id\']} (INT4 양자화) ...")\n'
     '    quant_cfg = BitsAndBytesConfig(',
     '    print(f"[로딩] {cfg[\'id\']} (INT4 양자화) ...")\n'
     '    import time as _time_load\n'
     '    _load_t0 = _time_load.perf_counter()\n'
     '    quant_cfg = BitsAndBytesConfig('),

    ("로딩 타이머 종료 + 출력",
     "    model.eval()\n"
     "    return model, processor, cfg",
     '    model.eval()\n'
     '    _load_dt = _time_load.perf_counter() - _load_t0\n'
     '    print(f"[로딩 완료] 모델 로딩/초기화 {_load_dt:.1f}s (오버헤드; 첫 실행은 다운로드 포함)")\n'
     "    return model, processor, cfg"),
]


def main():
    if not TARGET.exists():
        sys.exit(f"[중단] {TARGET} 가 없습니다. vlm_bench/vlm_bench/ 안에서 실행하세요.")
    src = TARGET.read_text(encoding="utf-8")

    if "_load_t0" in src:
        print("[건너뜀] 이미 로딩 타이머가 있습니다.")
        return

    for desc, old, _ in PATCHES:
        if old not in src:
            print(f"[중단] '{desc}' 대상 문자열을 찾지 못했습니다. bench_vlm.py 가 예상과 다릅니다.")
            print("       파일을 수정하지 않았습니다. 이 메시지를 그대로 전달해주세요.")
            sys.exit(1)

    shutil.copy2(TARGET, TARGET.with_suffix(".py.bak3"))
    for desc, old, new in PATCHES:
        src = src.replace(old, new, 1)
        print(f"[적용] {desc}")
    TARGET.write_text(src, encoding="utf-8")

    try:
        py_compile.compile(str(TARGET), doraise=True)
    except py_compile.PyCompileError as e:
        print("[경고] 문법 오류. 백업(bench_vlm.py.bak3)으로 되돌리세요.")
        print(e)
        sys.exit(1)

    print("\n[완료] 로딩 시간 측정 추가. 백업: bench_vlm.py.bak3")
    print("이제 실행 시 콘솔에 '[로딩 완료] 모델 로딩/초기화 X.Xs' 가 찍힙니다.")


if __name__ == "__main__":
    main()
