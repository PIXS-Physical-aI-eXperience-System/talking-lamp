"""파트 C(음성) 런타임 모듈.

벤치마크 스크립트(bench/, runners/)와 달리 이쪽이 제품에 들어가는 코드다.

  agent.py   마이크 프레임을 받아 언제 듣고 언제 말할지 정한다 (장치 없음)
  link.py    ROS 노드와의 메시지 규약
  stt.py     faster-whisper
  tts.py     MeloTTS ONNX, 문장 단위 분할 합성
  wake.py    웨이크워드 "픽스야"
  vad.py     발화 감지
  audio.py   표본율 변환 (장치 입출력은 파이가 맡으므로 쓰지 않는다)

띄우는 방법은 ros/README.md 참고.
"""
from .agent import IDLE, LISTENING, SPEAKING, THINKING, VoiceAgent  # noqa: F401
