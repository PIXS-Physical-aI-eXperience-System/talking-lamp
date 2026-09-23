#!/usr/bin/env bash
# Jetson 환경 점검 — 설치 전에 무엇이 있고 없는지부터 확인한다.
# 설치는 바꾸지 않는다. 읽기만 한다.
echo "================ Jetson 환경 점검 ================"

echo "── 보드/JetPack"
[ -f /etc/nv_tegra_release ] && cat /etc/nv_tegra_release || echo "  ! /etc/nv_tegra_release 없음 — Jetson이 아닐 수 있다"
[ -f /proc/device-tree/model ] && echo "  모델: $(tr -d '\0' < /proc/device-tree/model)"
command -v nvcc >/dev/null && echo "  CUDA: $(nvcc --version | grep release)" || echo "  CUDA: nvcc 없음 (런타임만 있을 수 있음)"

echo "── OS/파이썬"
grep PRETTY_NAME /etc/os-release 2>/dev/null | sed 's/^/  /'
echo "  아키텍처: $(uname -m)"
for v in 3.8 3.9 3.10 3.11 3.12; do
  p=$(command -v python$v) && echo "  python$v → $p"
done

echo "── 메모리"
free -h 2>/dev/null | sed 's/^/  /'
echo "  스왑: $(swapon --show=NAME,SIZE --noheadings 2>/dev/null | tr '\n' ' ')"

echo "── 전력 모드 (성능에 크게 영향)"
command -v nvpmodel >/dev/null && sudo -n nvpmodel -q 2>/dev/null | sed 's/^/  /' || echo "  nvpmodel 조회 불가 (sudo 필요)"
command -v jetson_clocks >/dev/null && echo "  jetson_clocks 있음 — 측정 전 실행 권장" || echo "  jetson_clocks 없음"

echo "── 파이썬 휠 가용성 (설치하지 않고 확인만)"
PY=$(command -v python3.10 || command -v python3)
for pkg in onnxruntime onnxruntime-gpu ctranslate2 faster-whisper transformers g2pkk python-mecab-ko; do
  if $PY -m pip index versions "$pkg" >/dev/null 2>&1; then
    echo "  ✔ $pkg"
  else
    echo "  ✗ $pkg — aarch64 휠 없음 가능. 소스 빌드 필요할 수 있다"
  fi
done

echo "── 필요한 산출물"
D="$(cd "$(dirname "$0")/.." && pwd)"
for f in models/melo-ko-onnx/melo-ko-vits.int8.onnx models/melo-ko-onnx/bert-kor-base.int8.onnx \
         models/melo-ko-onnx/frontend.json models/melo-ko-onnx/tokenizer/tokenizer_config.json; do
  [ -e "$D/$f" ] && echo "  ✔ $f" || echo "  ✗ $f — 맥에서 rsync 로 옮길 것"
done
n=$(ls "$D"/ref/*.wav 2>/dev/null | wc -l | tr -d ' ')
[ "$n" -gt 0 ] && echo "  ✔ ref/*.wav ($n개)" || echo "  ✗ ref/*.wav — 맥에서 rsync 로 옮길 것 (CER 비교에 필요)"
echo "=================================================="
