# InternVL3.5-2B RAM 경량화 PC 실측

2026-09-10, `feat/ram-lightweight`. **PC 도커에서 모델을 실제 실행한 결과이며 Jetson 실측/에뮬레이션 결과가 아니다.**

## 환경과 입력

- RTX 5070 Ti 16GB, x86_64, NVIDIA driver 595.84, 전력 상한 300W (변경하지 않음).
- 기존 이미지 `deeplearning-dl-gpu:latest`, image ID `sha256:fa502f7421587ef952b8b250faa461e439a7570d79f1c81ed6e5e6467b92b64e`.
- Python 3.12, torch 2.8.0+cu128, torchvision 0.23.0+cu128, transformers 4.57.6, bitsandbytes 0.48.2, accelerate 1.12.0. 기존 젯슨 문서의 라이브러리 버전과 다르다.
- `OpenGVLab/InternVL3_5-2B-HF`, revision `3f301ffcf3dcbb47893afae6650ea3e78d96fb6d`, 기존 벤치와 같은 nf4 + double quant + fp16 compute.
- [공개 이미지 australia.jpg](https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/transformers/tasks/australia.jpg), 1300×876. 원본 젯슨 테스트 사진은 확보하지 못했다. SHA256은 각 JSON에 기록했다.
- 모델 다운로드를 먼저 완료한 뒤 `--network none` 컨테이너에서 실행. 각 조건은 새 프로세스, 모델 로딩 1회 → 첫 추론 1회 → warm 5회 → 객체 삭제 및 CUDA 캐시 반환 → 프로세스 종료.
- 모든 결과 단위 GB = 10^9 bytes. RSS는 20ms, 시스템 메모리는 50ms 샘플링. GPU 수치는 PyTorch tensor allocated 피크이며 드라이버/전체 VRAM 점유가 아니다.

## 결과

| 조건 | 타일 / 비전 토큰 | 출력 토큰(실제) | 추론 CUDA 피크 GB | 추론 CPU RSS 피크 GB | 로딩 CPU RSS 피크 GB | warm 평균 초 |
| --- | --- | --- | --- | --- | --- | --- |
| 기본, 한국어 최대 128 | 7 / 1792 | 128 | 2.483 | 2.085 | 5.741 | 2.821 |
| 1타일, 한국어 최대 128 | 1 / 256 | 128 | 2.220 | 2.055 | 5.736 | 2.805 |
| 1타일, 한국어 최대 32 | 1 / 256 | 32 | 2.220 | 2.056 | 5.739 | 0.696 |
| 1타일, 영어 한 문장 최대 32 | 1 / 256 | 18 | 2.220 | 2.066 | 5.753 | 0.411 |

- **1타일 제한으로 추론 CUDA 피크 0.262GB, 10.6% 감소.** 같은 128토큰 출력에서는 속도 차이가 작았다.
- 32토큰 상한은 이 입력에서 피크를 추가로 줄이지 않았다. 생성 시간은 줄었으나 출력량도 함께 줄어든 결과다. 동일 품질/동일 작업량 속도 향상으로 해석하지 않는다.
- 모델 로딩은 2.88~3.06초. 첫 추론은 조건 순으로 3.417 / 3.222 / 1.107 / 0.830초. 다운로드 시간은 제외했고 디스크 캐시는 통제하지 않았다.
- **로딩 RSS 피크 약 5.74GB는 그대로다.** RSS에는 파일 매핑/공유 페이지도 포함하므로 전부 회수 불가능한 RAM은 아니다. CUDA와 RSS를 단순 합산해 젯슨 요구량으로 보고하지 않는다.
- 객체 삭제 + gc + empty_cache 이후에도 CPU RSS 약 2.05~2.08GB, CUDA allocated 약 8.5MB가 남았다. 이는 객체/allocator/프레임워크 상주가 남을 수 있음을 보여주며, 프로세스 종료를 별도로 검증해야 한다. CUDA 컨텍스트 메모리는 allocated 지표에 포함되지 않는다.
- 네 프로세스 모두 종료 코드 0, 측정 구간의 호스트 swap in/out 증가 0. 통합 스택 7.2GB 게이트의 통과 판정은 아니다.

## 적용한 변경과 미적용 항목

`tools/bench_vlm_ram.py`에서 processor의 `crop_to_patches=False`, `min_patches=1`, `max_patches=1`을 설정하고 실제 pixel_values 타일 수가 1인지 assert한다. max_new_tokens=32와 영어 단문 프롬프트는 별도 실험 조건이다. 체크포인트 설정 파일의 crop_to_patches 값만으로 실행 시 타일 수를 추정하면 안 된다. 이 환경의 기본 실제 입력은 7타일이었다.

단계 E(1B 모델)는 아래에 측정했다. STT/TTS 및 통합 모델 매니저는 **아직 미적용**이다. AWQ/TurboMind(단계 D)는 `docs/benchmarks/pc-ram-2026-09-10/lmdeploy-turbomind.md`에 호환성 결과를 별도로 기록한다.

## 단계 E: InternVL3.5-1B 비교

같은 환경·이미지·프로파일로 `OpenGVLab/InternVL3_5-1B-HF`(revision `main`, 동일 nf4 설정)를 측정했다. 2B와 나란히 비교하며, 절대 수치는 위 2B 표와 같은 방식으로 읽는다.

| 조건 | 타일 / 비전 토큰 | 출력 토큰(실제) | 추론 CUDA 피크 GB | 추론 CPU RSS 피크 GB | 로딩 CPU RSS 피크 GB | 로딩 CUDA 피크 GB | warm 평균 초 |
| --- | --- | --- | --- | --- | --- | --- | --- |
| 기본, 한국어 최대 128 | 7 / 1792 | 128 | 1.302 | 2.125 | 3.103 | 1.023 | 3.136 |
| 1타일, 한국어 최대 128 | 1 / 256 | 128 | 1.073 | 2.067 | 3.100 | 1.023 | 2.948 |
| 1타일, 한국어 최대 32 | 1 / 256 | 32 | 1.070 | 2.053 | 3.106 | 1.023 | 0.704 |
| 1타일, 영어 한 문장 최대 32 | 1 / 256 | 23 | 1.070 | 2.054 | 3.130 | 1.023 | 0.497 |

- **1B는 2B 대비 추론 CUDA 피크가 조건별 약 1.15~1.18GB 낮다** (1타일 128토큰 기준 2.220 → 1.073GB). 로딩 CUDA 피크도 약 1.02GB로 2B보다 낮다.
- **로딩 CPU RSS 피크가 5.74GB → 약 3.10GB로 약 2.6GB 낮다.** 이번 측정에서 RSS 로딩 피크 축소가 CUDA 축소보다 크다.
- 1타일 제한은 1B에서도 추론 CUDA 피크를 약 0.23GB(17%) 낮췄다. 32토큰 상한은 추가 피크 절감이 없다(2B와 동일 경향).
- 로딩 시간 `load_s`는 7.5~7.9초로 2B(2.9~3.1초)보다 길게 나왔는데, 이 프로세스에서 모델을 방금 내려받아 디스크 캐시가 warm하지 않았다. 재측정 필요. 첫 추론은 3.25 / 3.60 / 1.12 / 0.99초.
- 측정 프로세스 4개 모두 종료 코드 0.

### 1B 품질 (동일 단일 이미지)

2B보다 뚜렷하게 나쁘다.

- 한국어 128토큰(기본·1타일 모두): 한자 혼입(`중華门`), 같은 구절 반복(`한 개의 마을을 보여주는 한 개의 마을`), 근거 없는 고유명사(`부동산 시티`, `KUO`) 생성. 2B도 한국·한자 혼입은 있었으나 1B는 문장 구조가 더 무너진다.
- 영어 한 문장: `"A street scene shows a stop sign and a black car driving past a traditional Chinese gate with red lanterns."` 주요 물체(정지 표지판·검은 차·중식 문)와 일치. 이 한 장에서는 2B 영어 출력과 비슷한 수준.
- 결론: 메모리는 확실히 줄지만 한국어 서술 품질이 제품 기준에 못 미친다. 책상 장면·다중 이미지 평가 없이 1B 채택을 결정하지 않는다.

원자료는 `1b/` 하위 폴더의 `{baseline,one_tile,short_ko,short_en}.json` 및 `.log`.

## 품질 제한

기본과 1타일 한국어 출력 모두 사진을 한국이라고 단정하고 한자 등이 섞였다. 한국어 32토큰은 내용이 거의 시작되기 전에 끝나므로 제품용 답변으로 부적합하다. 영어 출력은 "A stop sign stands in front of a Chinese gate with a black car passing by."로 보이는 주요 물체와 일치하지만, 단일 이미지 관찰일 뿐이다. 책상 장면·한국어 대화 품질을 통과했다고 할 수 없다.

## 재실행 및 원자료

저장소 루트에서 준비된 CUDA Python 환경을 사용한다. 의존성은 `tools/requirements-vlm-ram.txt`, 설치 순서는 `docs/RAM-경량화-설계.md` 참고.

```bash
bash tools/run_vlm_ram_profiles.sh /path/to/australia.jpg results/pc-ram
```

동일 폴더의 `{baseline,one_tile,short_ko,short_en}.json`에는 입력 shape, revision, 각 실행 응답·시간·GPU 메모리·RSS 시계열을 보관했다. `*-system.json`은 호스트 전체 MemAvailable 기반 압박과 swap 기록이며 컨테이너 전용 RAM 측정이 아니다. `.log`에는 실행 stdout/stderr가 있다. 1B 결과는 `1b/`, TurboMind 결과는 `lmdeploy/` 및 `lmdeploy-turbomind.md`.

```bash
# 1B (단계 E): 같은 venv, --model-id만 교체
python tools/bench_vlm_ram.py --image australia.jpg --model-id OpenGVLab/InternVL3_5-1B-HF --out 1b/baseline.json

# TurboMind (단계 D): 격리 venv, tools/requirements-vlm-lmdeploy.txt 참고
python tools/bench_vlm_lmdeploy.py --image australia.jpg --cache-max-entry-count 0.1 --out lmdeploy/turbomind_hf_c0_1.json
```

PC/Jetson의 메모리 구조 차이: [NVIDIA CUDA for Tegra](https://docs.nvidia.com/cuda/cuda-for-tegra-appnote/index.html). 도커의 메모리 제한은 이 하드웨어 차이를 없애지 않는다.
