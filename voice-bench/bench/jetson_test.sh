#!/usr/bin/env bash
# Jetson에서 STT·TTS를 실측한다.
#
#   ./bench/jetson_test.sh setup    환경 구축 (한 번)
#   ./bench/jetson_test.sh stt      STT 정확도·메모리
#   ./bench/jetson_test.sh tts      TTS int8/fp32 비교
#   ./bench/jetson_test.sh soak     100 사이클 x 3회 (예산 근거)
#   ./bench/jetson_test.sh all
#
# 맥에서 얻은 수치와 비교하려면 ref/*.wav 가 같은 파일이어야 한다.
set -uo pipefail
cd "$(dirname "$0")/.."
ROOT="$PWD"
PY=$(command -v python3.10 || command -v python3)
V="$ROOT/venvs/melo-onnx"

log() { echo; echo "── $*"; }

setup() {
  log "venv 생성 ($PY)"
  [ -d "$V" ] || "$PY" -m venv "$V"
  "$V/bin/pip" install -q --upgrade pip

  log "의존성 (onnxruntime 은 마지막에 깐다)"
  "$V/bin/pip" install -q numpy soundfile transformers num2words jamo anyascii g2p_en g2pkk python-mecab-ko onnx || {
    echo "  ! 일부 실패. 위 오류 확인"; }

  log "faster-whisper (STT)"
  # 주의: faster-whisper 는 의존성으로 CPU판 onnxruntime 을 끌고 온다.
  # 그래서 반드시 이걸 먼저 깔고, onnxruntime-gpu 를 나중에 덮어써야 한다.
  # 순서를 바꾸면 CPU판이 GPU판을 덮어써서 CUDAExecutionProvider 가 사라진다.
  "$V/bin/pip" install -q faster-whisper || {
    echo "  ! 실패 — ctranslate2 의 aarch64 휠이 없을 수 있다. 소스 빌드 필요"; }

  log "onnxruntime — GPU 빌드로 마무리"
  "$V/bin/pip" uninstall -qy onnxruntime onnxruntime-gpu 2>/dev/null
  if ! "$V/bin/pip" install -q onnxruntime-gpu 2>/dev/null; then
    echo "  onnxruntime-gpu 실패 → CPU판으로 대체"
    echo "  ※ GPU로 돌리려면 NVIDIA의 JetPack용 휠이 필요할 수 있다:"
    echo "    https://developer.download.nvidia.com/compute/redist/jp/"
    "$V/bin/pip" install -q onnxruntime
  fi

  log "설치 결과"
  "$V/bin/python" - <<'PY'
import importlib
for m in ("onnxruntime", "faster_whisper", "transformers", "numpy", "soundfile"):
    try:
        mod = importlib.import_module(m)
        print(f"  ✔ {m} {getattr(mod,'__version__','')}")
    except Exception as e:
        print(f"  ✗ {m}: {type(e).__name__}")
try:
    import onnxruntime as ort
    provs = ort.get_available_providers()
    print("  실행 공급자:", provs)
    if "CUDAExecutionProvider" not in provs:
        print("  ! CUDA 공급자가 없다 — CPU로만 돈다.")
        print("    onnxruntime 과 onnxruntime-gpu 가 같이 깔려 있지 않은지 확인:")
        print("      venvs/melo-onnx/bin/pip list | grep -i onnxruntime")
except Exception:
    pass
try:
    import torch; print(f"  ! torch가 설치돼 있다 ({torch.__version__}) — 이 파이프라인의 전제는 torch 없음")
except ImportError:
    print("  ✔ torch 미설치 (의도된 상태)")
PY
}

stt() {
  log "STT — faster-whisper small, CUDA 시도 후 실패 시 CPU"
  for dev in cuda cpu; do
    ct=$([ "$dev" = cuda ] && echo int8_float16 || echo int8)
    echo "  [$dev / $ct]"
    out=$(mktemp)
    TOKENIZERS_PARALLELISM=false "$V/bin/python" bench/runners/stt_faster_whisper.py \
      --model small --device "$dev" --compute-type "$ct" \
      --ref-dir "$ROOT/ref" --label "fw-small-$dev" >"$out" 2>&1
    if ! grep -q '@@RESULT@@' "$out"; then
      echo "    실패 — 아래는 실제 출력이다:"; sed 's/^/      /' "$out" | tail -25; rm -f "$out"; continue
    fi
    grep -o '@@RESULT@@.*' "$out" | "$V/bin/python" -c "
import json,sys
s=sys.stdin.read().replace('@@RESULT@@','')
d=json.loads(s)
print(f\"    피크 RSS {d['peak_rss_mb']:.0f} MB  로드 {d['load_s']}s\")
for t in d['transcripts']: print('    ·', t)
"
    rm -f "$out"; break
  done
}

tts() {
  for mode in int8 fp32; do
    flag=$([ "$mode" = int8 ] && echo --int8 || echo "")
    log "TTS — $mode"
    out=$(mktemp)
    TOKENIZERS_PARALLELISM=false "$V/bin/python" runners/tts_melo_onnx.py \
      --out-dir "$ROOT/out/tts/jetson-$mode" --label "melo-onnx-$mode" --normalize $flag >"$out" 2>&1
    if ! grep -q '@@RESULT@@' "$out"; then
      echo "    실패 — 아래는 실제 출력이다:"; sed 's/^/      /' "$out" | tail -25; rm -f "$out"; continue
    fi
    grep -o '@@RESULT@@.*' "$out" | "$V/bin/python" -c "
import json,sys
s=sys.stdin.read().replace('@@RESULT@@','')
d=json.loads(s)
rtf=[a/b for a,b in zip(d['synth_s'], d['audio_s'])]
print(f\"    피크 RSS {d['peak_rss_mb']:.0f} MB   RTF {sum(rtf)/len(rtf):.3f} (문장별 {min(rtf):.2f}~{max(rtf):.2f})\")
print(f\"    공급자 {d['config']['providers']}\")
print('    ※ RTF가 1.0을 넘으면 실시간보다 느려 대화에 못 쓴다' if sum(rtf)/len(rtf) > 1 else '')
"
    rm -f "$out"
  done
}

soak() {
  for mode in melo-onnx-int8 melo-onnx; do
    log "내구 측정 — $mode, 100 사이클 x 3회"
    for i in 1 2 3; do
      TOKENIZERS_PARALLELISM=false "$V/bin/python" bench/soak.py "$mode" 100 >/dev/null 2>&1
      "$V/bin/python" -c "
import csv,sys
try:
    v=[float(r[1]) for r in list(csv.reader(open('$ROOT/out/soak-$mode.tsv'),delimiter='\t'))[1:]]
    print(f'    {$i}회차  최고 {max(v):.0f} MB  평균 {sum(v)/len(v):.0f} MB')
except Exception as e: print('    실패:', e)"
    done
  done
  echo
  echo "  ※ 예산은 평균이 아니라 3회 중 최고값으로 잡는다."
}

case "${1:-all}" in
  setup) setup ;;
  stt)   stt ;;
  tts)   tts ;;
  soak)  soak ;;
  all)   setup; stt; tts; soak ;;
  *) echo "사용법: $0 [setup|stt|tts|soak|all]"; exit 1 ;;
esac
