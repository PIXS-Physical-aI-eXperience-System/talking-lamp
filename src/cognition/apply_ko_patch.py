#!/usr/bin/env python3
"""bench_vlm.py 자동 패치 — 혼입 억제(--suppress-cjk) + 경량화(--img-max-side) 추가.

vlm_bench/vlm_bench/ 폴더(bench_vlm.py 옆)에 두고 한 번 실행:
    python3 apply_ko_patch.py

하는 일:
  - bench_vlm.py 를 bench_vlm.py.bak 으로 백업
  - 아래 5곳을 문자열 치환으로 수정 (ko_lite.py import, 이미지 축소, 디코딩 억제, CLI 2개)
  - 각 치환이 정확히 매칭되는지 확인하고, 하나라도 안 맞으면 아무것도 안 바꾸고 중단
  - 끝나면 문법 검사(py_compile)까지
다시 실행하면 '이미 패치됨'으로 안전하게 건너뜀.
"""
import pathlib
import py_compile
import shutil
import sys

TARGET = pathlib.Path("bench_vlm.py")

# (설명, old, new) — old 는 파일에 정확히 1회 이상 나와야 함
PATCHES = [
    ("import 추가",
     "from common import MemMonitor",
     "from common import MemMonitor\n"
     "from ko_lite import resize_max_side, get_ko_logits_processors, sanitize_for_tts, contains_foreign"),

    ("ask() 시그니처에 옵션 추가",
     "korean_only: bool = True) -> tuple[str, float]:",
     "korean_only: bool = True, suppress_cjk: bool = False,\n"
     "         img_max_side: Optional[int] = None) -> tuple[str, float]:"),

    ("경량화: 이미지 축소 삽입",
     '    img = Image.open(image_path).convert("RGB")\n'
     '    text = (KOREAN_ONLY_INSTRUCTION',
     '    img = Image.open(image_path).convert("RGB")\n'
     '    img = resize_max_side(img, img_max_side)   # 경량화: 타일 수 감소 -> 메모리/속도 감소\n'
     '    text = (KOREAN_ONLY_INSTRUCTION'),

    ("혼입 억제: generate 옵션 추가",
     "    t0 = time.perf_counter()\n"
     "    with torch.no_grad():\n"
     "        out = model.generate(**inputs, max_new_tokens=max_new_tokens, do_sample=False)",
     '    gen_kwargs = dict(max_new_tokens=max_new_tokens, do_sample=False)\n'
     '    if suppress_cjk:   # 혼입 억제: 한자/가나 토큰 차단 + 반복 루프 방지\n'
     '        gen_kwargs["logits_processor"] = get_ko_logits_processors(processor.tokenizer)\n'
     '        gen_kwargs["no_repeat_ngram_size"] = 3\n'
     '        gen_kwargs["repetition_penalty"] = 1.1\n'
     '\n'
     "    t0 = time.perf_counter()\n"
     "    with torch.no_grad():\n"
     "        out = model.generate(**inputs, **gen_kwargs)"),

    ("CLI 옵션 2개 추가",
     'help="한국어 전용 system 프롬프트를 끄고 원래 방식으로 실행 (비교용)")',
     'help="한국어 전용 system 프롬프트를 끄고 원래 방식으로 실행 (비교용)")\n'
     '    ap.add_argument("--suppress-cjk", action="store_true",\n'
     '                    help="한자/가나 토큰을 디코딩 단계에서 차단 (혼입 억제 + 반복 방지)")\n'
     '    ap.add_argument("--img-max-side", type=int, default=None,\n'
     '                    help="이미지 긴 변을 이 px로 축소 (경량화; 448이면 대략 1타일=최소 메모리)")'),

    ("ask() 호출에 인자 전달",
     "answer, elapsed = ask(model, processor, img_path, q, args.max_new_tokens, korean_only)",
     "answer, elapsed = ask(model, processor, img_path, q, args.max_new_tokens,\n"
     "                                      korean_only, args.suppress_cjk, args.img_max_side)"),
]


def main():
    if not TARGET.exists():
        sys.exit(f"[중단] {TARGET} 가 현재 폴더에 없습니다. vlm_bench/vlm_bench/ 안에서 실행하세요.")

    src = TARGET.read_text(encoding="utf-8")

    if "from ko_lite import" in src:
        print("[건너뜀] 이미 패치되어 있습니다 (from ko_lite import 발견).")
        return

    # 1) 먼저 전부 매칭되는지 검사 (하나라도 안 맞으면 아무것도 안 함)
    for desc, old, _ in PATCHES:
        if old not in src:
            print(f"[중단] '{desc}' 대상 문자열을 찾지 못했습니다. bench_vlm.py 가 예상과 다릅니다.")
            print("       파일을 수정하지 않았습니다. 이 메시지를 그대로 전달해주세요.")
            sys.exit(1)

    # 2) 백업 후 치환
    shutil.copy2(TARGET, TARGET.with_suffix(".py.bak"))
    for desc, old, new in PATCHES:
        src = src.replace(old, new, 1)
        print(f"[적용] {desc}")

    TARGET.write_text(src, encoding="utf-8")

    # 3) 문법 검사
    try:
        py_compile.compile(str(TARGET), doraise=True)
    except py_compile.PyCompileError as e:
        print("[경고] 패치 후 문법 오류가 있습니다. 백업(bench_vlm.py.bak)으로 되돌리세요.")
        print(e)
        sys.exit(1)

    print("\n[완료] bench_vlm.py 패치 성공. 백업: bench_vlm.py.bak")
    print("이제 아래 옵션을 쓸 수 있습니다:")
    print("  --suppress-cjk         혼입 억제 (한자/가나 차단 + 반복 방지)")
    print("  --img-max-side 448     경량화 (이미지 축소 -> 타일/메모리 감소)")


if __name__ == "__main__":
    main()
