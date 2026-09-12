# RAM 경량화 설계 및 측정 계획

상태: **Jetson 최종 게이트 통과.** 1B NF4·1타일·16토큰·닫힌 집합 한국어
렌더러·상태별 직렬 실행을 3회 실측했다. 최악 시스템 피크 4.146GB, 최소 여유
3.703GB, swap 0이며 VLM→한국어→TTS→STT 왕복 품질도 3회 통과했다.
`docs/benchmarks/jetson-ram-2026-09-12/` 참고.
기준: main `34c183d`, 브랜치 `feat/ram-lightweight`. PC 실측 결과는 `docs/benchmarks/pc-ram-2026-09-10/`.

## 근거와 범위

- 저장소 밖 `../notes/2026-09-09_VLM-스택-경량화.md`를 설계 입력으로 사용한다.
- `docs/vlm-benchmark`의 `docs/VLM-벤치마크-결과.md`, `src/cognition/mem_breakdown.py`, `add_loadtimer.py`, `add_qwen.py`를 확인했다. 해당 브랜치에는 원본 `vlm_bench/bench_vlm.py`와 실험 이미지가 없다. 원본을 확인하기 전 문자열 패치를 만들어 적용하지 않는다.
- 기존 결과는 InternVL3.5-2B 29.7초, 시스템 피크 약 6.1GB다. 새 브랜치에서 재현한 값이 아니다. 원본 측정 코드 일부는 1024 기반 환산을 GB로 표기하므로 새 기록은 bytes 원본과 GB(10^9)/GiB(2^30)를 구분한다.
- 상위 문서의 2~3초, 4.1~5.6GB 및 로딩 2~4초는 가설이다. 부품별 추정 합을 통합 실측으로 보고하지 않는다.

## 적용 순서

| 단계 | 변경 | 검증 |
| --- | --- | --- |
| A | 기존 2B/nf4 조건 재현 | 동일 이미지·질문·토큰 상한 및 환경 기록 |
| B | 실제 processor에서 비전 타일 1개로 제한 | 텐서 shape·실제 타일/비전 토큰 수 기록. 단순 이미지 축소와 구분 |
| C | 영어 한 문장, 출력 상한 32 토큰 | 128 토큰 조건과 각각 비교, 실제 생성 토큰 수와 응답 원문 저장 |
| D | 같은 2B의 AWQ/TurboMind 후보 | **PC 부분 완료**: lmdeploy 0.17.0 TurboMind는 `InternVL3_5-2B-HF`만 변환됨(원본 형식 실패). fp16 전용, warm 추론 약 0.96초(bnb 3배 빠름)이나 GPU 7.3GB+, 변환 시 CPU RSS 21.5GB로 Jetson RAM 목표에 역행. AWQ 자가 양자화·사전 변환·aarch64 빌드 미확인. `docs/benchmarks/pc-ram-2026-09-10/lmdeploy-turbomind.md` |
| E | 1B 후보 | 동일 질문으로 장면 인식·누락·환각 평가, 2B 대비 비교. **완료**: 추론 CUDA 피크 약 1.07GB(2B 2.22GB 대비 -1.15GB), 로딩 RSS 약 3.1GB(2B 5.74GB 대비 -2.6GB). 한국어 서술 품질은 2B보다 뚜렷이 낮음. `docs/benchmarks/pc-ram-2026-09-10/README.md` 단계 E |
| F | 상태별 모델 프로세스 종료/재기동 | **완료**: STT→VLM→TTS 직렬 실행. 최종 전체 벤치 42.22~51.21초, 종료 후 메모리 반환 확인 |

각 변경을 하나씩 비교한 뒤 최종 조합을 측정한다. 첫 실행 다운로드는 별도 준비 단계로 분리하고, 오프라인 실행 가능 여부도 확인한다.

LMDeploy는 AWQ와 InternVL 배포 경로를 제공하지만, 이것만으로 현재 Jetson 환경과 특정 InternVL3.5 HF 체크포인트의 호환성이 확정되지는 않는다. 설치 버전과 모델별 지원을 먼저 확인한다.
공식 참고: [InternVL 배포](https://lmdeploy.readthedocs.io/en/stable/multi_modal/internvl.html), [설치 안내](https://github.com/InternLM/lmdeploy/blob/main/docs/en/get_started/installation.md).

## 상태별 RAM 소유권

| 상태 | 상주 | 무거운 작업 |
| --- | --- | --- |
| idle / S1 / S4 / S6 | 오케스트레이터, 호출어, 검출·기하 | VLM·STT·TTS 워커 없음 |
| listening | 위 구성 + STT 워커 | 음성 텍스트 확정 후 STT 종료 |
| thinking | 위 기본 구성 + VLM 워커 | 이미지 1장·bounded queue, 결과 확정 후 워커 종료 |
| speaking | 위 기본 구성 + TTS 워커 | 발화 완료 후 TTS 종료 |

프로세스 종료로 CUDA 컨텍스트까지 반환한다. `empty_cache()`만 호출하는 것을 언로드 완료로 취급하지 않는다. 전환은 이전 워커 종료 확인 후 다음 워커를 시작한다. 동시에 두 대화 요청을 실행하지 않고 오래된 카메라 프레임은 교체한다. 타임아웃·취소·실패 시에도 워커 회수가 필요하다.

이 정책의 RAM 이득과 매 턴 로딩 지연은 함께 평가한다. 항상 언로드할지, 제한 시간 동안 유지할지는 측정 후 결정한다. 현재 main에 STT/TTS/VLM 통합 구현이 없으므로 이 표는 통합 계약이며 적용 완료 상태가 아니다. 번역 모델 추가 시 별도 상태와 메모리 측정이 필요하다.

## 측정

`tools/ram_probe.py`는 외부 벤치 프로세스를 실행하며 시작 전, 로딩과 추론 전체, 종료 후 시스템 RAM과 swap I/O를 기록한다. torch를 가져오지 않아 프레임워크 초기화도 포함한다.

```bash
python3 tools/ram_probe.py --out results/baseline-system.json -- \
  /path/to/vlm-venv/bin/python /path/to/vlm_bench/bench_vlm.py \
  --model internvl --images-dir /path/to/test_images --limit 1
```

위 모델 alias와 벤치 인자는 원본 확인 후 확정한다. 측정 도구는 임의의 명령을 `--` 뒤에 받을 수 있다. 벤치 stdout은 그대로 표시되고 실패 코드는 전달된다.

- 매 조건은 새 프로세스에서 시작한다. 모델을 한 번 로드한 뒤 첫 추론과 warm 반복 5회를 별도로 기록한다.
- 원본 벤치 내부에서는 CUDA synchronize 후 로딩/추론 시간, 실제 출력 토큰, allocated/reserved 피크를 기록한다. 시스템 RAM과 CUDA 메모리를 더하면 안 된다.
- 시스템 `MemTotal - MemAvailable`은 전체 메모리 압박의 근사이며 프로세스 RSS가 아니다. 다른 작업, 캐시 변화, 짧은 샘플 사이 피크의 영향을 받는다. 원본 `mem_breakdown.py`의 시작 대비 증분과 절대 시스템 점유를 구분한다.
- 모델 ID/revision, 라이브러리 버전, 보드 모델, 전력 모드, 온도, 입력 이미지 해시, 프롬프트, 타일 수, 출력 길이를 결과와 함께 보관한다.
- 이미지 EXIF 보정과 동일 입력을 유지한다. 일반 장면과 작은 글자·가림이 있는 장면의 응답 원문을 품질 검토한다.
- 합격은 통합 실행 전체 피크 7.2GB 이하, swap 증가와 swap I/O 없음, 응답 품질 유지로 판정한다. VLM 단독 측정만으로 통합 게이트 통과라고 하지 않는다.

## 현재 결과

Jetson 실측 없음. 보드와 원본 벤치 입력이 확보되면 A의 원래 조건 재현을 진행한다. OS 데몬 종료·전력 모드 변경·기존 Python 환경 교체는 수행하지 않았다.

전체 bench_vlm.py 없이도 기존 브랜치의 독립 실행 코드 mem_breakdown.py를 참고해 비교 실험은 가능하다. `tools/bench_vlm_ram.py`와 `tools/run_vlm_ram_profiles.sh`가 그 구현이다. 각 프로필을 별도 프로세스로 실행하고 첫 추론 1회와 warm 5회, 로딩/추론 RSS, CUDA allocated/reserved, 실제 타일·토큰 수, 출력 원문을 저장한다. 가중치는 기존과 같은 nf4이며 AWQ 교체는 아직 적용하지 않았다.

## PC 도커 실험의 의미

5070 Ti에서 x86 CUDA 컨테이너를 사용한다. 도커로 Python/CUDA 사용자 공간을 격리하고 오프라인 실행을 확인할 수 있지만, Jetson의 CPU/GPU 공유 DRAM이나 성능을 에뮬레이션하지 않는다. `--memory=8g`만으로 Jetson 8GB를 재현할 수도 없다. PC에서는 호스트 RAM과 VRAM이 분리되어 있으므로 각각 보고한다.

근거: [NVIDIA CUDA for Tegra 메모리 구조](https://docs.nvidia.com/cuda/cuda-for-tegra-appnote/index.html), [Jetson 컨테이너의 x86 실행 제약](https://nvidia.github.io/container-wiki/toolkit/jetson.html).

환경 준비 예시(기존 torch 2.8.0+cu128 컨테이너 내부):

```bash
python -m venv --system-site-packages /work/venv
/work/venv/bin/pip install --no-deps torchvision==0.23.0 --index-url https://download.pytorch.org/whl/cu128
/work/venv/bin/pip install -r tools/requirements-vlm-ram.txt
```

모델 다운로드 후 `PATH`에 위 venv를 우선 설정하고 실행한다. 컨테이너에 호스트 UID를 쓰는 경우 `USER` 환경변수도 설정한다.

```bash
bash tools/run_vlm_ram_profiles.sh /path/to/image.jpg results/pc-ram
```

원본 실험 이미지가 없으므로 공개 샘플로 먼저 메모리 비교만 한다. 동일 타일 수가 확인되면 1타일 설정 효과를 0으로 보고하고 임의의 절감치를 주장하지 않는다.
