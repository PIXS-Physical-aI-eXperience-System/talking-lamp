# 온보드 LLM 벤치마크 결과 (A파트 · 인지 · 주경태)

**목적:** VLM(InternVL/Qwen2-VL)에서 **텍스트 LLM으로 전환** 후, 온보드에서 돌릴 소형 한국어 LLM을 선정한다.
**결론:** **HyperCLOVA X SEED 1.5B 선정.**
**상태:** LLM 선정·프롬프트·파서·검출(D) 연계까지 완료. 음성(C) 연결·파서 통합은 팀 협의 필요.

---

## 0. 측정 환경

| 항목 | 값 |
|---|---|
| 보드 | Jetson Orin Nano 8GB (가용 RAM 7.3GB) |
| 스택 | JetPack 7.2 / L4T r39 / CUDA 13.2 / Python 3.12 |
| 서빙 | **ollama 네이티브 설치**(GPU). ⚠️ JetPack 7용 ollama 컨테이너가 없어서 도커는 CPU 폴백 → 네이티브 설치로 GPU 확보 |
| 양자화 | GGUF Q4_K_M |
| 모델 저장 | NVMe(`/mnt/ssd/ollama`) — eMMC 용량 부족으로 |

> GPU 확인: `ollama ps` → `100% GPU`. 네이티브 ollama가 드라이버 하위호환으로 CUDA 13.2에서 GPU 구동됨.

---

## 1. 후보 3종 (전부 GPU 구동)

| 모델 | 크기 | 라이선스 | 비고 |
|---|---|---|---|
| Qwen2.5-3B-Instruct | 3B | Apache 2.0 | 저번 VLM 때 영어용으로 썼던 계열 |
| Kanana 2.1B (Kakao) | 2.1B | 허용적 | 한국어 특화 |
| **HyperCLOVA X SEED 1.5B (Naver)** | 1.5B | HyperCLOVA X SEED 약관 | **한국어 네이티브, 텍스트 최대 크기(3B는 Vision 전용)** |

---

## 2. 결과 (GPU + few-shot + temp 0.2 기준)

| 지표 | Qwen2.5-3B | Kanana 2.1B | **HyperCLOVA 1.5B** |
|---|---|---|---|
| 모델 메모리 | ~2.2GB | ~2.4GB | **~2.0GB** ✅ |
| 속도 tok/s | 12~14 | 15~17 | **24** ✅ |
| TTFT | 0.10s | 0.11s | 0.11s |
| **한국어 자연스러움** | 약함(영어샘·딱딱) | 좋음 | ✅ **최고** |
| **JSON 행동태그 준수** | 80~93% | ❌ 0% (형식 불안정) | ✅ **100%** |
| 환각 | 있음 | 있음 | ✅ 없음(튜닝 후) |

---

## 3. 선정: HyperCLOVA X SEED 1.5B

**이유 (선정 기준 = "용량 낮으면서 한국어 성능 좋은 것"):**
1. **한국어가 제일 자연스러움** — 램프 대화가 핵심인데 셋 중 말맛이 최고.
2. **제일 가볍고 빠름** (2.0GB / 24 tok/s) — STT·TTS·검출기와 예산(7.2GB) 공유하므로 경량이 유리.
3. **프롬프트 튜닝 후 JSON 100% + 환각 없음.**
4. Qwen은 라이선스는 깨끗하나 한국어 약함, Kanana는 JSON을 못 맞춰 탈락.

**참고:** HyperCLOVA 텍스트 모델은 **0.5B/1.5B만** 존재(3B는 Vision-Instruct=VLM). 1.5B가 텍스트 최대. 라이선스 약관 최종 확인 예정.

---

## 4. 프롬프트 엔지니어링 (반복 기록)

| 버전 | 설정 | JSON 준수 | 태그 | 환각 |
|---|---|---|---|---|
| v1 | few-shot 없음 | 73% | ❌ 안 씀 | 있음 |
| v2 | few-shot, temp 0.7 | 67~87% | △ 엉뚱 | 줄음 |
| **v3** | few-shot+접지 규칙, **temp 0.2** | ✅ **100%** | ✅ 대체로 맞음 | ✅ 없음 |

**핵심 조치:** ①행동태그 사용 예시(few-shot) ②"사실 지어내지 마라" 접지 규칙 ③temperature 0.2로 형식 안정화.

---

## 5. 산출물 (이 브랜치)

```
src/cognition/
  cognition_parser.py      행동태그 JSON 파서 (깨진 출력 방어)
  cognition_run.py         LLM+파서 통합 모듈 — respond(labels, text) → {say, actions}
  llm_bench.py             벤치마크 도구 (속도·메모리·지시준수 자동 측정)
  vision_to_cognition.py   A↔D 브릿지 (이미지→YOLOX→한국어 라벨→인지)
  live_cognition.py        라이브 카메라 E2E
docs/
  LLM-벤치마크-결과.md      이 문서
  행동태그-스키마-v1.md      say/actions JSON 계약 (B·E 사인오프 대기)
  프롬프트-엔지니어링-v1.md   분류라벨→대사 프롬프트 설계
```

---

## 6. 검출기(D) 연계 — 실제 이미지 E2E 확인

`vision_to_cognition.py`로 실제 책상 사진(h029) 관통 확인:
```
검출: 물병, 키보드, 의자, 마우스, 노트북
→ "내 책상 지금 어때?" → say: "정리정돈이 잘 되어 있네요." + happy_wiggle
```
- 라벨을 나열하지 않고 자연스럽게 반영, 없는 것 안 지어냄, 태그 적절.
- **검출 정확도(예: 개→고양이 오분류)는 D 영역**, A는 "주어진 라벨로 잘 판단"이 목표 → 달성.
- D 검출기는 **영어 COCO 라벨** 출력 → 한국어 매핑은 현재 A 브릿지에서 처리(인터페이스 합의 필요).

---

## 7. 남은 일 (팀 협의)

- **음성(C/최승원) 연결:** sw107 `voice/agent.py --llm`이 우리 ollama 엔드포인트(`127.0.0.1:11434/v1/chat/completions`)에 바로 붙는 구조. 마이크·STT·TTS 환경이 필요해 **최승원과 함께** 진행.
- **파서 통합:** sw107은 현재 평문 스트리밍을 기대. 우리 JSON+파서를 어디서 돌릴지 **C·B와 협의.**
- **행동태그 스키마 사인오프:** B(김태현)·E(이수혁) 확인.
- **라이선스:** HyperCLOVA X SEED 약관 최종 확인.
