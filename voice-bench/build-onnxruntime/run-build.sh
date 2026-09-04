#!/usr/bin/env bash
# 컨테이너 안에서 실제 컴파일을 수행한다.
# 빌드 디렉터리(/build)는 볼륨이므로, 실패 후 다시 실행하면 cmake 가
# 이미 만든 오브젝트를 재사용해 이어서 진행한다.
set -euo pipefail
PARALLEL="${PARALLEL:-4}"
NVCC_THREADS="${NVCC_THREADS:-2}"
ARCH="${CUDA_ARCH:-87}"

echo "== onnxruntime 빌드 시작 (sm_${ARCH}, 병렬 ${PARALLEL} x nvcc ${NVCC_THREADS}) =="

# flash attention / memory efficient attention 커널을 뺀다.
# 3차 빌드가 이 파일들에서 컨테이너 메모리 상한에 걸려 죽었다:
#   flash_fwd_split_hdim128_fp16_causal_sm80.cu
#   cicc 하나가 2.4 GB 를 썼고, 병렬 9 면 상한 20 GB 를 넘는다.
# 트랜스포머 어텐션 최적화 커널이라 우리 모델(VITS·BERT, 표준 opset 17)에는
# 쓰이지 않는다. 공식 문서도 빌드 시간 단축 수단으로 이 두 옵션을 안내한다.
# contrib_ops 전체를 빼는 것과 달리 이건 지원되는 옵션이라 링크가 깨지지 않는다.
#
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
      onnxruntime_BUILD_UNIT_TESTS=OFF \
      onnxruntime_USE_FLASH_ATTENTION=OFF \
      onnxruntime_USE_MEMORY_EFFICIENT_ATTENTION=OFF

echo "== 휠 복사 =="
cp /build/Release/dist/*.whl /out/
ls -lh /out/
