#!/usr/bin/env bash
# Jetson Orin(sm_87)용 ctranslate2 휠을 빌드한다.
#
#   ./build.sh          환경 이미지 준비 후 빌드 (실패하면 다시 실행 — 이어서 진행된다)
#   ./build.sh shell    컨테이너 안으로 들어가 직접 확인
#
# 구조는 build-onnxruntime/ 과 같다. 빌드 디렉터리를 도커 볼륨에 두므로
# 중간에 깨져도 다시 실행하면 이어서 간다. onnxruntime(3시간)보다는 작지만
# qemu 위에서 도는 건 마찬가지라 넉넉히 잡을 것.
set -euo pipefail
cd "$(dirname "$0")"
mkdir -p dist

HOST_ARCH=$(uname -m)
if [ -r /proc/meminfo ]; then
  MEM_GB=$(( $(awk '/MemAvailable/{print $2}' /proc/meminfo) / 1024 / 1024 )); CORES=$(nproc)
else
  MEM_GB=$(( $(docker info --format '{{.MemTotal}}') / 1024 / 1024 / 1024 )); CORES=$(docker info --format '{{.NCPU}}')
fi
# nvcc(cicc)는 커널당 수 GB 를 쓰고 qemu 위에서는 더 쓴다. onnxruntime 때
# 병렬 13 으로 돌렸다가 OOM 킬러가 호스트 systemd 까지 죽여 데스크탑이
# 재부팅됐다. 4 GB 당 1 개로 잡고 상한을 건다.
PARALLEL="${PARALLEL:-$(( MEM_GB / 4 ))}"
[ "$PARALLEL" -gt "$CORES" ] && PARALLEL=$CORES
[ "$PARALLEL" -lt 2 ] && PARALLEL=2
MEM_LIMIT=$(( MEM_GB * 3 / 4 )); [ "$MEM_LIMIT" -lt 4 ] && MEM_LIMIT=4

echo "호스트 $HOST_ARCH | 가용 메모리 ${MEM_GB}GB | 코어 $CORES → 병렬도 $PARALLEL, 컨테이너 상한 ${MEM_LIMIT}GB"
if [ "$HOST_ARCH" != "aarch64" ] && [ "$HOST_ARCH" != "arm64" ]; then
  echo "aarch64 가 아니므로 qemu 에뮬레이션으로 빌드한다 (느리다. 밤새 걸어둘 것)"
  docker run --privileged --rm tonistiigi/binfmt --install arm64 >/dev/null 2>&1 || true
fi

echo "── 환경 이미지 준비 (첫 회만 오래 걸린다. 이후는 캐시)"
docker build --platform linux/arm64 -t ct2-jetson-sm87 .

VOL=ct2-build-cuda130
docker volume create "$VOL" >/dev/null

if [ "${1:-}" = "shell" ]; then
  exec docker run --rm -it --platform linux/arm64 \
    --memory "${MEM_LIMIT}g" --memory-swap "${MEM_LIMIT}g" \
    -v "$VOL":/build -v "$PWD/dist:/out" --entrypoint bash ct2-jetson-sm87
fi

echo "── 컴파일 (실패해도 다시 실행하면 이어서 진행)"
docker run --rm --platform linux/arm64 \
  --memory "${MEM_LIMIT}g" --memory-swap "${MEM_LIMIT}g" \
  -e PARALLEL="$PARALLEL" -e CUDA_ARCH_LIST=8.7 \
  -v "$VOL":/build -v "$PWD/dist:/out" \
  ct2-jetson-sm87

echo
echo "완료. Jetson 으로 옮겨 설치:"
echo "  scp dist/*.whl <jetson>:~/"
echo "  cd ~/talking-lamp/voice-bench"
echo "  venvs/melo-onnx/bin/pip install --force-reinstall ~/ctranslate2-*.whl"
echo "  venvs/melo-onnx/bin/python -c \"import ctranslate2; print(ctranslate2.get_cuda_device_count())\""
