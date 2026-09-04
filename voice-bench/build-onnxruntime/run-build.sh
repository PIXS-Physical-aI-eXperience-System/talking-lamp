#!/usr/bin/env bash
# 컨테이너 안에서 실제 컴파일을 수행한다.
# 빌드 디렉터리(/build)는 볼륨이므로, 실패 후 다시 실행하면 cmake 가
# 이미 만든 오브젝트를 재사용해 이어서 진행한다.
set -euo pipefail
PARALLEL="${PARALLEL:-4}"
ARCH="${CUDA_ARCH:-87}"

echo "== onnxruntime 빌드 시작 (sm_${ARCH}, 병렬 ${PARALLEL}) =="

# --disable_contrib_ops:
#   onnxruntime 1.29 의 contrib_ops CUDA 커널이 CUDA 13.2 의 CCCL 헤더와
#   충돌해 컴파일이 깨진다(device_transform.cuh, proclaims_copyable_arguments).
#   이건 onnxruntime + CUDA 13 의 알려진 비호환이며 우리 설정 문제가 아니다.
#   우리 모델(melo VITS·한국어 BERT)은 표준 opset 17 연산만 쓰고 contrib
#   도메인 연산이 하나도 없음을 확인했으므로 빼도 무방하다.
#   덤으로 빌드 시간도 크게 줄어든다 — 실패 직전 95~98% 구간이 전부 contrib 였다.
./build.sh \
  --build_dir /build \
  --config Release \
  --use_cuda --cuda_home "$CUDA_HOME" --cudnn_home /usr \
  --build_wheel --skip_tests --allow_running_as_root \
  --disable_contrib_ops \
  --parallel "$PARALLEL" \
  --cmake_extra_defines \
      CMAKE_CUDA_ARCHITECTURES="$ARCH" \
      onnxruntime_BUILD_UNIT_TESTS=OFF

echo "== 휠 복사 =="
cp /build/Release/dist/*.whl /out/
ls -lh /out/
