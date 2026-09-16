"""파트 C(음성) 런타임 모듈.

벤치마크 스크립트(bench/, runners/)와 달리 이쪽이 제품에 들어가는 코드다.
B 의 런타임 골격에서 VoicePipeline 을 만들어 쓰면 된다.
"""
from .pipeline import IDLE, LISTENING, SPEAKING, THINKING, VoicePipeline  # noqa: F401
