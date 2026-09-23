# melo_text — MeloTTS 텍스트 프론트엔드 사본 (torch 없음)

MeloTTS 패키지는 import 시 torch를 끌고 온다. 그런데 한국어 텍스트 처리
(정규화 → 자모 → 음소 → 심볼 ID)는 실제로 torch를 전혀 쓰지 않는다.

ONNX 파이프라인에서 torch를 완전히 제거하기 위해 필요한 파일만 복사해 왔다.
- 언어별 `*_bert.py`는 torch를 쓰므로 제외 (BERT 특징은 ONNX로 따로 계산한다)
- 쓰지 않는 언어(중국어/일본어/프랑스어/스페인어) 모듈도 제외
- `korean.py`의 `from melo.text.ko_dictionary` → 상대 import 로 변경

원본: MeloTTS (MIT) / myshell-ai
