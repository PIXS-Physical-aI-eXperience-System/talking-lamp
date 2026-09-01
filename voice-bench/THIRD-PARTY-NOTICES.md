# 제3자 구성요소 고지

파트 C(음성)의 한국어 TTS 파이프라인이 사용하는 외부 저작물과, 우리가 가한 변경 사항.

Apache-2.0 §4(b)는 변경한 파일에 변경 사실을 명시할 것을 요구하고,
MIT·Apache-2.0 모두 저작권 고지와 라이선스 사본의 유지를 요구한다.
아래가 그 고지다.

---

## 1. MeloTTS — 코드

- **출처**: https://github.com/myshell-ai/MeloTTS
- **저작권**: MyShell.ai
- **라이선스**: MIT
- **사용 범위**: 한국어 텍스트 프론트엔드(정규화 → 자모 → 음소 → 심볼 ID)

**우리가 가한 변경** (`melo_text/`):

1. 필요한 파일만 복사 — `__init__.py`, `symbols.py`, `cleaner.py`, `korean.py`,
   `english.py`, `ko_dictionary.py`. 나머지 언어 모듈과 `*_bert.py`는 제외했다
   (`*_bert.py`가 torch를 끌어오는데, ONNX 파이프라인에서는 torch를 쓰지 않는다).
2. `cleaner.py` — 7개 언어를 한 줄로 전부 import 하던 것을 지연 로드로 변경.
   원본은 일본어 모듈이 `mecab-python3`(모듈명 `MeCab`)를, 한국어 모듈이
   `python-mecab-ko`(모듈명 `mecab`)를 요구하는데, 대소문자를 구분하지 않는
   파일시스템(macOS)에서 두 패키지가 공존할 수 없다.
3. `english.py` — 일본어 모듈에서 가져오던 `distribute_phone`을 파일 안에 옮겨 정의.
   의존성 없는 순수 함수이며, 위와 같은 이유로 일본어 모듈 로드를 피하기 위함이다.
4. `korean.py` — `from melo.text.ko_dictionary import`를 상대 import로 변경.

## 2. MeloTTS-Korean — 음성 합성 모델 가중치

- **출처**: https://huggingface.co/myshell-ai/MeloTTS-Korean
- **라이선스**: MIT
  - 모델 카드 원문: "This library is under MIT License, which means it is free for
    both commercial and non-commercial use."

**우리가 가한 변경**:

1. PyTorch 체크포인트를 **ONNX로 변환** (`melo_export_onnx.py`)
   → `melo-ko-vits.onnx` (170.5 MB)
2. 변환본을 **int8 동적 양자화** (`melo_quantize.py`)
   → `melo-ko-vits.int8.onnx` (53.4 MB)

가중치의 수치 표현이 바뀌었을 뿐 학습된 모델 자체는 원본과 같다.

**미확인 사항**: 이 모델의 학습에 사용된 음성 데이터의 출처와 화자는
모델 카드에 공개되어 있지 않다. 배포자가 MIT로 배포했으나, 목소리의 출처를
우리가 확인한 바는 없다는 점을 기록해 둔다.

## 3. bert-kor-base — 한국어 BERT

- **출처(가중치)**: https://huggingface.co/kykim/bert-kor-base
- **출처(원 저장소)**: https://github.com/kiyoungkim1/LMkor
- **저작권**: Kiyoung Kim
- **라이선스**: **Apache-2.0**
  - LMkor README 원문: "The pretrained models is distributed under the terms of
    the Apache-2.0 License."
  - 전문: https://www.apache.org/licenses/LICENSE-2.0

> **주의**: HuggingFace 모델 카드 자체에는 라이선스 표기가 없다. Apache-2.0은
> 모델 카드가 링크하는 GitHub 원 저장소에 명시되어 있다. 근거가 한 단계 건너뛰어
> 있으므로 출처를 원 저장소로 기록한다.

**우리가 가한 변경** (Apache-2.0 §4(b)에 따른 고지):

1. `AutoModelForMaskedLM` 대신 `AutoModel`로 로드해 **MLM 헤드를 제외**했다.
   파이프라인은 `hidden_states[-3]`만 사용하며 MLM 헤드는 쓰지 않는다.
   출력은 원본과 동일하다(코사인 유사도 1.000000 실측).
2. 위 구성을 **ONNX로 변환** (`melo_export_bert.py`)
   → `bert-kor-base.onnx` (414.3 MB)
3. 변환본을 **int8 동적 양자화** (`melo_quantize.py`)
   → `bert-kor-base.int8.onnx` (104.2 MB)

**상업적 사용에 관한 참고**: LMkor README에는 "모델의 상업적 사용의 경우 MOU를
통해 무료로 사용하실 수 있습니다"라는 안내와 연락처(kykim@artificial.sc)가 있다.
Apache-2.0 자체는 상업적 사용을 허용하므로 법적 강제 조건은 아니나, 제품화·유상
배포로 나아갈 경우 저자에게 연락하는 것이 안전하다. 학내 프로젝트·공모전 출품
범위에서는 해당하지 않는다.

---

## 배포 시 지켜야 할 것

`models/melo-ko-onnx/` 산출물을 저장소 밖으로 배포(시연 이미지, 릴리스 등)할 때는
이 파일과 함께 다음을 포함한다.

- MIT 라이선스 전문 및 MyShell.ai 저작권 고지 (MeloTTS 코드·가중치)
- Apache-2.0 라이선스 전문 및 Kiyoung Kim 저작권 고지 (BERT)

모델 가중치 자체는 용량 때문에 저장소에 커밋하지 않는다.
`melo_export_onnx.py` → `melo_export_bert.py` → `melo_quantize.py` 순서로
실행하면 동일한 산출물을 재생성할 수 있다.
