# Jetson Orin(sm_87)용 ctranslate2 휠 빌드 — 환경 이미지.
#
# 왜 필요한가:
#   PyPI 의 ctranslate2 aarch64 휠은 CUDA 없이 빌드돼 있다. Jetson 에서
#   device="cuda" 를 주면 이렇게 죽는다:
#     ValueError: This CTranslate2 package was not compiled with CUDA support
#   CPU 로는 faster-whisper small 이 RTF 1.07 (6코어 전부 사용) 이라 실시간이
#   안 된다. 정확도는 CER 0.000 으로 완벽하니, 모델이 아니라 실행 경로를 고친다.
#
# 대상 환경: Ubuntu 24.04 / Python 3.12 / CUDA 13.x sbsa / cuDNN 9 / aarch64
FROM arm64v8/ubuntu:24.04

# CUDA 13.2 는 onnxruntime 빌드에서 CCCL 헤더 충돌을 냈다. 여기서도 같은
# 문제가 날 수 있어 검증된 13.0 으로 맞춘다. NVIDIA 의 마이너 버전 호환
# 규칙에 따라 13.2 런타임(Jetson)에서 동작한다 — onnxruntime 으로 확인된 사실이다.
ARG CUDA_PKG=13-0
# CUDA 13 에서 컴파일되는지는 확인된 바 없다. 실패하면 이 값을 올려서 다시 시도할 것.
ARG CT2_VERSION=v4.6.0
ENV DEBIAN_FRONTEND=noninteractive

RUN apt-get update && apt-get install -y --no-install-recommends \
      build-essential cmake git wget ca-certificates \
      python3.12 python3.12-dev python3-pip python3-venv \
      libopenblas-dev \
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
      numpy packaging setuptools wheel pybind11

RUN git clone --recursive --depth 1 --branch ${CT2_VERSION} \
      https://github.com/OpenNMT/CTranslate2 /src/CTranslate2

COPY run-build.sh /usr/local/bin/run-build.sh
RUN chmod +x /usr/local/bin/run-build.sh
WORKDIR /src/CTranslate2
ENTRYPOINT ["/usr/local/bin/run-build.sh"]
