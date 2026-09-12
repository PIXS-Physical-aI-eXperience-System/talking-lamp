# 단계 D: LMDeploy TurboMind 호환성·메모리 (PC)

2026-09-10, `feat/ram-lightweight`. **PC(x86_64, RTX 5070 Ti, Blackwell sm_120) 결과다. Jetson(aarch64, sm_87) 호환성은 여기서 확인되지 않는다.**

목표: InternVL3.5-2B를 bitsandbytes nf4 대신 LMDeploy TurboMind로 돌렸을 때의 설치 호환성과 GPU/RAM 사용량 비교.

## 설치 호환성

- `pip install lmdeploy==0.17.0`을 그대로 하면 의존성 해석이 **torch 2.12.1+cu130 / transformers 5.17 / nvidia-\*-cu13**으로 승격되고, 사전 빌드된 `_turbomind.so`가 `libcublas.so.12`(CUDA 12)를 찾다 실패한다 (`ImportError: libcublas.so.12`).
- 제약 파일로 `torch==2.8.0+cu128`, `torchvision==0.23.0+cu128`, `transformers==4.57.6`, `triton==3.4.0`을 고정하면 설치가 통과하고 `from lmdeploy.turbomind import turbomind`가 import된다. (`--gpus all`로 `libcuda.so.1` 필요.)
- 격리 venv: `5070ti:/work/venv-lmdeploy2` (`--system-site-packages`, 컨테이너 `deeplearning-dl-gpu:latest`).
- **aarch64에서는 `lmdeploy` PyPI wheel에 `triton`/`tilelang`/`apache-tvm-ffi` 의존성이 marker로 빠지고, Jetson은 소스 빌드가 필요하다. 이 실험은 그 경로를 확인하지 않았다.**

## 체크포인트 형식

| 체크포인트 | TurboMind 변환 | 결과 |
| --- | --- | --- |
| `OpenGVLab/InternVL3_5-2B` (원본) | 실패 | 변환기가 Qwen3 텍스트 백본 설정을 dict로 전달, `AttributeError: 'dict' object has no attribute 'hidden_size'` (`lmdeploy/turbomind/models/qwen3.py`). |
| `OpenGVLab/InternVL3_5-2B-HF` (transformers 형식) | 성공 | 로딩·추론 동작. 아래 측정. |

InternVL3.5-2B는 Qwen3-1.7B 텍스트 백본이다. lmdeploy 0.17.0에서 원본 InternVL3.5 형식은 변환 불가, **`-HF` 변형만 동작**한다.

## 측정 (`-HF` + TurboMind)

- `OpenGVLab/InternVL3_5-2B-HF`, `session_len 8192`, 7타일 입력(비전+텍스트 약 1812 토큰), 한국어 128토큰, warm 5회. GPU는 `nvidia-smi` 전체 used, RSS는 psutil 20~50ms 샘플. GB = 10^9 bytes.
- **TurboMind는 이 경로에서 4bit 양자화를 쓰지 않는다. fp16 가중치(약 4.4GB)를 적재한다.** InternVL3.5-2B용 공개 AWQ 체크포인트가 없어 `lmdeploy lite auto_awq` 자가 양자화(보정 데이터 필요)는 이번에 하지 않았다.

| cache_max_entry_count | GPU 로딩 후 GB | GPU 피크 GB | 로딩 CPU RSS 피크 GB | 추론 CPU RSS 피크 GB | 언로드 후 GPU GB | 로딩 초 | warm 평균 초 | 출력 토큰 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 0.1 | 7.27 | 7.47 | 21.53 | 21.69 | 0.85 | 13.0 | 0.959 | 128 |
| 0.5 | 11.66 | 11.67 | 21.53 | 21.69 | 0.85 | 12.0 | 0.961 | 128 |

- **추론 속도는 warm 약 0.96초로 bnb nf4(2B, 7타일, 2.821초)보다 약 3배 빠르다.** 첫 추론도 약 1.0초(bnb 3.4초).
- **GPU 사용량은 bnb nf4보다 크다.** bnb 벤치의 torch allocated 피크 2.48GB는 CUDA 컨텍스트·reserved·비전 인코더를 제외한 수치라 직접 비교는 아니지만, TurboMind는 fp16 가중치 + KV 풀로 `cache 0.1`에서도 전체 7.3GB를 점유한다. `cache 0.5`는 11.7GB. Jetson 8GB 통합 메모리에는 `cache`를 매우 낮춰도 빠듯하다.
- **치명적 지점: HF 체크포인트 변환 때문에 CPU RSS가 약 21.5GB까지 올라간다.** TurboMind가 fp16 전체 가중치를 CPU에서 로드·재배치한 뒤 GPU로 올린다. Jetson 8GB에서는 이 상태로 로드 자체가 불가능하다. 회피하려면 `lmdeploy convert`로 사전 변환한 TurboMind 모델을 배포해야 하고, 이번엔 확인하지 않았다.
- 품질은 bnb 2B와 같은 수준이다. `"중화門"`, 한국 단정, 한자·깨진 토큰(`펜alty`, `두隻`) 혼입, 환각. TurboMind로 바꿔서 품질이 좋아지지는 않는다.
- 언로드 후 GPU 0.85GB(CUDA 컨텍스트), RSS 1.51GB 잔류. 완전 회수는 프로세스 종료 필요.

## 판단

PC에서 TurboMind `-HF` 경로는 **속도 이득은 크지만(약 3배) Jetson RAM 목표에는 역행한다**: fp16 전용, GPU 7.3GB+, 변환 시 CPU RSS 21.5GB. 다음이 선행되어야 후보로 유지 가능하다.

1. `lmdeploy lite auto_awq`로 2B AWX(W4A16) 자가 양자화 + 품질 재평가
2. `lmdeploy convert`로 사전 변환 모델 생성 (로딩 시 21GB RSS 회피 확인)
3. aarch64/Jetson에서 lmdeploy 소스 빌드 및 sm_87 커널 확인
4. AWQ 적용 후 GPU/RSS 재측정, 통합 7.2GB 게이트 대조

현재까지 확인된 결론: **2B/nf4 + 1타일 제한이 여전히 가장 안전한 기준선**이고, TurboMind는 위 4단계를 마치기 전에는 Jetson 적용 후보로 확정하지 않는다.

## 참고

- TurboMind는 자체 CUDA allocator와 KV 캐시 풀을 미리 확보한다. `cache_max_entry_count`가 여유 VRAM의 몇 %를 캐시에 예약할지 정한다. GPU 수치는 `nvidia-smi`의 전체 used memory(다른 부하 없는 GPU)이며 bnb 벤치의 torch allocated 피크와 직접 비교하면 안 된다. 같은 표에 `cache_max_entry_count`를 함께 적는다.
- 기본 `session_len` 4096은 이미지 토큰(7타일 ≈ 3841)에 KV 예약이 겹쳐 warm-up 경고가 난다. `session_len 8192`로 측정했다.
- 원자료: `5070ti:/work/results-lmdeploy/` (repo에는 `lmdeploy/` 하위로 복사).
- 공식 문서: [InternVL 배포](https://lmdeploy.readthedocs.io/en/stable/multi_modal/internvl.html), [설치](https://github.com/InternLM/lmdeploy/blob/main/docs/en/get_started/installation.md).
