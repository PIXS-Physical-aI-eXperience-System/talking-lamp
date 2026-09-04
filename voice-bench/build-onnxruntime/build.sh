#!/usr/bin/env bash
# Jetson Orin(sm_87)용 onnxruntime 휠을 빌드한다.
#
#   ./build.sh          환경 이미지 준비 후 빌드 (실패하면 다시 실행 — 이어서 진행된다)
#   ./build.sh shell    컨테이너 안으로 들어가 직접 확인
#
# 빌드 디렉터리를 도커 볼륨(ort-build)에 두므로, 컴파일이 중간에 깨져도
# 다시 실행하면 cmake 가 만들어둔 오브젝트를 재사용한다. 한 번에 3시간이
# 걸리는 작업이라 이 구조가 아니면 재시도 비용이 감당되지 않는다.
set -euo pipefail
cd "$(dirname "$0")"
mkdir -p dist

HOST_ARCH=$(uname -m)
if [ -r /proc/meminfo ]; then
  MEM_GB=$(( $(awk '/MemAvailable/{print $2}' /proc/meminfo) / 1024 / 1024 )); CORES=$(nproc)
else
  MEM_GB=$(( $(docker info --format '{{.MemTotal}}') / 1024 / 1024 / 1024 )); CORES=$(docker info --format '{{.NCPU}}')
fi
# nvcc(cicc)는 커널당 수 GB 를 쓰고, qemu 위에서는 더 쓴다.
# 메모리 27 GB 에서 병렬 13 으로 돌렸다가 cicc 가 OOM 으로 죽었다.
# 4 GB 당 1 개, 최대 8 개로 제한한다.
PARALLEL=$(( MEM_GB / 4 )); [ "$PARALLEL" -gt "$CORES" ] && PARALLEL=$CORES
[ "$PARALLEL" -gt 8 ] && PARALLEL=8
[ "$PARALLEL" -lt 2 ] && PARALLEL=2

# 컨테이너 메모리 상한. 도커는 기본적으로 제한이 없어서 컴파일러가 호스트
# 메모리를 전부 먹을 수 있다. 실제로 첫 시도에서 OOM 킬러가 systemd 까지
# 죽여 데스크탑이 재부팅됐다. 상한을 걸면 넘칠 때 컨테이너 안의 프로세스만
# 죽고 호스트는 멀쩡하다 — 빌드는 다시 이어서 돌리면 된다.
MEM_LIMIT=$(( MEM_GB * 3 / 4 )); [ "$MEM_LIMIT" -lt 4 ] && MEM_LIMIT=4

echo "호스트 $HOST_ARCH | 가용 메모리 ${MEM_GB}GB | 코어 $CORES → 병렬도 $PARALLEL, 컨테이너 상한 ${MEM_LIMIT}GB"
if [ "$HOST_ARCH" != "aarch64" ] && [ "$HOST_ARCH" != "arm64" ]; then
  echo "aarch64 가 아니므로 qemu 에뮬레이션으로 빌드한다 (느리다. 밤새 걸어둘 것)"
  docker run --privileged --rm tonistiigi/binfmt --install arm64 >/dev/null 2>&1 || true
fi

echo "── 환경 이미지 준비 (첫 회만 오래 걸린다. 이후는 캐시)"
docker build --platform linux/arm64 -t ort-jetson-sm87 .

VOL=ort-build-cuda130   # CUDA 버전이 바뀌면 이 이름을 바꿔 새로 시작한다
docker volume create "$VOL" >/dev/null

if [ "${1:-}" = "shell" ]; then
  exec docker run --rm -it --platform linux/arm64 \
    --memory "${MEM_LIMIT}g" --memory-swap "${MEM_LIMIT}g" \
    -v "$VOL":/build -v "$PWD/dist:/out" --entrypoint bash ort-jetson-sm87
fi

echo "── 컴파일 (실패해도 다시 실행하면 이어서 진행)"
docker run --rm --platform linux/arm64 \
  --memory "${MEM_LIMIT}g" --memory-swap "${MEM_LIMIT}g" \
  -e PARALLEL="$PARALLEL" -e CUDA_ARCH=87 \
  -v "$VOL":/build -v "$PWD/dist:/out" \
  ort-jetson-sm87

echo
echo "완료. Jetson 으로 옮겨 설치:"
echo "  scp dist/*.whl <jetson>:~/"
echo "  cd ~/talking-lamp/voice-bench"
echo "  venvs/melo-onnx/bin/pip uninstall -y onnxruntime onnxruntime-gpu"
echo "  venvs/melo-onnx/bin/pip install ~/onnxruntime_gpu-*.whl"
