#!/usr/bin/env bash
# Jetson Orin(sm_87)용 onnxruntime 휠을 빌드한다.
#
# 왜 필요한가:
#   PyPI 의 onnxruntime-gpu 휠에는 Jetson 전용 아키텍처 sm_87 커널이 없어
#   Orin 에서 cudaErrorNoKernelImageForDevice 로 죽는다. 직접 빌드해야 한다.
#   JetPack 7(R39) 용 사전 빌드 휠은 jetson-ai-lab 에도 NVIDIA 배포처에도 없다.
#
# 어디서 빌드하나:
#   aarch64 휠이 필요하다. Apple Silicon 이나 ARM 서버면 네이티브로 빠르고,
#   x86_64 리눅스면 qemu 에뮬레이션이라 느리지만 방치할 수 있다.
#   빌드에 GPU 는 필요 없다 — nvcc 는 장치 없이도 sm_87 기계어를 만든다.
#
#   ./build.sh
#
# 결과물 dist/*.whl 을 Jetson 으로 옮겨 설치한다.
set -euo pipefail
cd "$(dirname "$0")"
mkdir -p dist

HOST_ARCH=$(uname -m)
# 램 1 GB 당 컴파일 작업 1 개를 넘기지 않는다 (nvcc 가 커널당 수 GB 를 쓴다).
if [ -r /proc/meminfo ]; then
  MEM_GB=$(( $(awk '/MemAvailable/{print $2}' /proc/meminfo) / 1024 / 1024 ))
  CORES=$(nproc)
else
  MEM_GB=$(( $(docker info --format '{{.MemTotal}}') / 1024 / 1024 / 1024 ))
  CORES=$(docker info --format '{{.NCPU}}')
fi
PARALLEL=$(( MEM_GB / 2 )); [ "$PARALLEL" -gt "$CORES" ] && PARALLEL=$CORES
[ "$PARALLEL" -lt 2 ] && PARALLEL=2

echo "호스트 $HOST_ARCH | 가용 메모리 ${MEM_GB}GB | 코어 $CORES → 병렬도 $PARALLEL"
if [ "$HOST_ARCH" != "aarch64" ] && [ "$HOST_ARCH" != "arm64" ]; then
  echo "aarch64 가 아니므로 qemu 에뮬레이션으로 빌드한다 (느리다. 밤새 걸어둘 것)"
  docker run --privileged --rm tonistiigi/binfmt --install arm64 >/dev/null 2>&1 || true
fi

echo "── 이미지 빌드 (수 시간~하룻밤)"
docker build --platform linux/arm64 --build-arg PARALLEL=$PARALLEL -t ort-jetson-sm87 .

echo "── 휠 꺼내기"
docker run --rm --platform linux/arm64 -v "$PWD/dist:/out" ort-jetson-sm87

echo
echo "완료. Jetson 으로 옮겨 설치:"
echo "  scp dist/*.whl <jetson>:~/"
echo "  cd ~/talking-lamp/voice-bench"
echo "  venvs/melo-onnx/bin/pip uninstall -y onnxruntime onnxruntime-gpu"
echo "  venvs/melo-onnx/bin/pip install ~/onnxruntime_gpu-*.whl"
