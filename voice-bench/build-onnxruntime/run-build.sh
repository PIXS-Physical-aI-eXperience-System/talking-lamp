#!/usr/bin/env bash
# 컨테이너 안에서 실제 컴파일을 수행한다.
# 빌드 디렉터리(/build)는 볼륨이므로, 실패 후 다시 실행하면 cmake 가
# 이미 만든 오브젝트를 재사용해 이어서 진행한다.
set -euo pipefail
PARALLEL="${PARALLEL:-4}"
NVCC_THREADS="${NVCC_THREADS:-2}"
ARCH="${CUDA_ARCH:-87}"

echo "== onnxruntime 빌드 시작 (sm_${ARCH}, 병렬 ${PARALLEL} x nvcc ${NVCC_THREADS}) =="

# contrib_ops 를 빼려 했으나 실패했다. 코어 CPU 커널(fp16_conv.cc)이
# contrib 에 있는 GetFusedActivationAttr 를 참조해서 링크가 깨진다
# (undefined reference). onnxruntime 자체가 코어→contrib 의존을 갖고 있어
# 선택적으로 제외할 수 없다. 대신 CUDA 를 13.0 으로 낮춰 충돌을 피한다.
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
