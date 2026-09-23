#!/usr/bin/env bash
# 컨테이너 안에서 실제 컴파일을 수행한다.
# 빌드 디렉터리(/build)는 볼륨이므로, 실패 후 다시 실행하면 cmake 가
# 이미 만든 오브젝트를 재사용해 이어서 진행한다.
set -euo pipefail
PARALLEL="${PARALLEL:-4}"
NVCC_THREADS="${NVCC_THREADS:-2}"
ARCH="${CUDA_ARCH:-87}"

echo "== onnxruntime 빌드 시작 (sm_${ARCH}, 병렬 ${PARALLEL} x nvcc ${NVCC_THREADS}) =="

# 구성요소를 빼려는 시도는 전부 실패했다. onnxruntime 은 코어가 선택적
# 구성요소를 참조하고 있어 깔끔하게 떼어지지 않는다:
#   --disable_contrib_ops        → 코어 링크 실패 (GetFusedActivationAttr)
#   USE_FLASH_ATTENTION=OFF      → 코어 컴파일 실패 (kCutlassSafeMaskFilterValue)
# 그래서 전부 기본값으로 두고, 메모리는 병렬도로만 조절한다.
#
# 1차 시도(전부 기본값, CUDA 13.2)는 98% 까지 갔고 유일한 실패 원인이
# CUDA 13.2 의 CCCL 헤더 충돌이었다. 그건 13.0 으로 해결됐다.
# 남은 위험은 flash attention 컴파일의 메모리 사용량뿐이며(cicc 하나당 2.4 GB),
# 병렬도를 낮추면 상한 안에 들어간다. PARALLEL=4 로 돌릴 것.
./build.sh \
  --build_dir /build \
  --config Release \
  --use_cuda --cuda_home "$CUDA_HOME" --cudnn_home /usr \
  --build_wheel --skip_tests --allow_running_as_root \
  --parallel "$PARALLEL" --nvcc_threads "$NVCC_THREADS" \
  --cmake_extra_defines \
      CMAKE_CUDA_ARCHITECTURES="$ARCH" \
      onnxruntime_BUILD_UNIT_TESTS=OFF

echo "== 휠 복사 =="
cp /build/Release/dist/*.whl /out/
ls -lh /out/
