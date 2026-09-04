# Jetson Orin(sm_87)용 onnxruntime-gpu 휠을 맥에서 빌드한다.
#
# 왜 맥에서 빌드하나:
#   PyPI 의 onnxruntime-gpu 휠에는 Jetson 전용 아키텍처인 sm_87 커널이 없어
#   Orin 에서 cudaErrorNoKernelImageForDevice 로 죽는다. 직접 빌드해야 한다.
#   그런데 Orin Nano 는 디스크 19 GB·램 8 GB·microSD 라 빌드가 버겁다.
#   Apple Silicon 도 aarch64 이므로 리눅스 컨테이너가 네이티브 속도로 돌고,
#   CUDA 코드 컴파일에는 GPU 가 필요 없다(nvcc 는 장치 없이도 sm_87 을 낸다).
#
# 대상 환경(Jetson)과 맞춰야 하는 것:
#   Ubuntu 24.04 / Python 3.12 / CUDA 13.2 (sbsa) / cuDNN 9 / aarch64
FROM arm64v8/ubuntu:24.04

ARG ORT_VERSION=v1.29.0
ARG CUDA_ARCH=87
# nvcc 는 커널당 수 GB 를 쓴다. 램 여유에 맞춰 넘겨줄 것 (build.sh 가 계산한다).
ARG PARALLEL=3
ENV DEBIAN_FRONTEND=noninteractive

RUN apt-get update && apt-get install -y --no-install-recommends \
      build-essential cmake git wget ca-certificates \
      python3.12 python3.12-dev python3-pip python3-venv \
      libpython3-dev protobuf-compiler libprotobuf-dev \
 && rm -rf /var/lib/apt/lists/*

# CUDA 13.2 + cuDNN 9 (sbsa = Jetson 이 쓰는 것과 같은 aarch64 서버 타깃)
RUN wget -q https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2404/sbsa/cuda-keyring_1.1-1_all.deb \
 && dpkg -i cuda-keyring_1.1-1_all.deb && rm cuda-keyring_1.1-1_all.deb \
 && apt-get update && apt-get install -y --no-install-recommends \
      cuda-toolkit-13-2 cudnn9-cuda-13 \
 && rm -rf /var/lib/apt/lists/*

ENV CUDA_HOME=/usr/local/cuda
ENV PATH=$CUDA_HOME/bin:$PATH
ENV LD_LIBRARY_PATH=$CUDA_HOME/lib64:$LD_LIBRARY_PATH

RUN pip3 install --break-system-packages --no-cache-dir \
      numpy packaging setuptools wheel

RUN git clone --recursive --depth 1 --branch ${ORT_VERSION} \
      https://github.com/microsoft/onnxruntime /src/onnxruntime

WORKDIR /src/onnxruntime
# --allow_running_as_root: 컨테이너는 root 로 돈다
# CMAKE_CUDA_ARCHITECTURES=87: 이 빌드의 존재 이유
RUN ./build.sh \
      --config Release \
      --use_cuda --cuda_home $CUDA_HOME --cudnn_home /usr \
      --build_wheel --skip_tests --allow_running_as_root \
      --parallel ${PARALLEL} \
      --cmake_extra_defines \
          CMAKE_CUDA_ARCHITECTURES=${CUDA_ARCH} \
          onnxruntime_BUILD_UNIT_TESTS=OFF

CMD ["bash", "-c", "cp /src/onnxruntime/build/Linux/Release/dist/*.whl /out/ && ls -lh /out/"]
