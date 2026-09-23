# Jetson Orin(sm_87)용 onnxruntime-gpu 휠 빌드 — 환경 이미지.
#
# 이 이미지는 "빌드할 수 있는 환경"까지만 만든다. 실제 컴파일은 build.sh 가
# docker run 으로 수행하며, 빌드 디렉터리를 볼륨에 두어 실패 후 재시도가
# 처음부터가 아니라 이어서 진행되게 한다 (한 번에 3시간이 걸린다).
#
# 왜 필요한가:
#   PyPI 의 onnxruntime-gpu 휠에는 Jetson 전용 아키텍처 sm_87 커널이 없어
#   Orin 에서 cudaErrorNoKernelImageForDevice 로 죽는다. JetPack 7(R39) 용
#   사전 빌드 휠은 어디에도 없다.
#
# 대상 환경과 맞춘 것: Ubuntu 24.04 / Python 3.12 / CUDA 13.x sbsa / cuDNN 9 / aarch64
FROM arm64v8/ubuntu:24.04

ARG ORT_VERSION=v1.29.0
# CUDA 13.2 의 CCCL 헤더가 onnxruntime 1.29 와 충돌한다
# (device_transform.cuh: global qualification of class name is invalid).
# 13.0 으로 빌드해도 NVIDIA 의 마이너 버전 호환 규칙에 따라 13.x 런타임에서
# 동작한다 — Jetson 은 13.2 지만 이 휠로 돌아간다.
ARG CUDA_PKG=13-0
ENV DEBIAN_FRONTEND=noninteractive

RUN apt-get update && apt-get install -y --no-install-recommends \
      build-essential cmake git wget ca-certificates \
      python3.12 python3.12-dev python3-pip python3-venv \
      libpython3-dev protobuf-compiler libprotobuf-dev \
 && rm -rf /var/lib/apt/lists/*

RUN wget -q https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2404/sbsa/cuda-keyring_1.1-1_all.deb \
 && dpkg -i cuda-keyring_1.1-1_all.deb && rm cuda-keyring_1.1-1_all.deb \
 && apt-get update && apt-get install -y --no-install-recommends \
      cuda-toolkit-${CUDA_PKG} cudnn9-cuda-13 \
 && rm -rf /var/lib/apt/lists/*

ENV CUDA_HOME=/usr/local/cuda
ENV PATH=$CUDA_HOME/bin:$PATH
ENV LD_LIBRARY_PATH=$CUDA_HOME/lib64:$LD_LIBRARY_PATH

RUN pip3 install --break-system-packages --no-cache-dir \
      numpy packaging setuptools wheel

RUN git clone --recursive --depth 1 --branch ${ORT_VERSION} \
      https://github.com/microsoft/onnxruntime /src/onnxruntime

COPY run-build.sh /usr/local/bin/run-build.sh
RUN chmod +x /usr/local/bin/run-build.sh
WORKDIR /src/onnxruntime
ENTRYPOINT ["/usr/local/bin/run-build.sh"]
