"""파이 ↔ 젯슨 사이의 메시지 규약.

TCP 위에 길이 접두 프레임을 얹는다. TCP 는 바이트 스트림이라 어디서 한 덩어리가
끝나는지 알려주지 않는다. 길이를 앞에 붙여야 받는 쪽이 자를 수 있다.

프레임: [4바이트 종류][4바이트 길이][본문]   (둘 다 빅엔디안)

오디오는 전부 16 kHz 모노 S16LE 다. 보드 펌웨어가 16 kHz 이고, 그보다 올려봐야
얻을 게 없다. GStreamer caps 로는 audio/x-raw,format=S16LE,rate=16000,channels=1.

왜 직접 만드는가: 전송은 B 가 GStreamer/TCP 로 하기로 했지만, 우리가 주고받는
것은 '연속 스트림' 이 아니라 '발화 한 덩어리' 와 '합성 결과' 다. 요청-응답이
있어야 하므로 그 위에 얇은 규약이 필요하다.
"""
import socket
import struct

import numpy as np

HDR = struct.Struct(">4sI")

# 파이 → 젯슨
STT_REQ = b"STTQ"     # 발화 오디오 (S16LE)
TTS_REQ = b"TTSQ"     # 합성할 문장 (UTF-8)
PING = b"PING"
# 젯슨 → 파이
STT_RES = b"STTR"     # 받아쓴 글 (UTF-8)
TTS_AUD = b"TTSA"     # 합성 오디오 조각 (S16LE). 여러 번 온다
TTS_END = b"TTSE"     # 합성 끝
ERROR = b"ERR!"
PONG = b"PONG"

RATE = 16000


def send(sock, kind, payload=b""):
    sock.sendall(HDR.pack(kind, len(payload)) + payload)


def _recv_exact(sock, n):
    buf = bytearray()
    while len(buf) < n:
        b = sock.recv(n - len(buf))
        if not b:
            raise ConnectionError("상대가 연결을 끊었다")
        buf += b
    return bytes(buf)


def recv(sock):
    """(종류, 본문). 연결이 끊기면 ConnectionError."""
    kind, n = HDR.unpack(_recv_exact(sock, HDR.size))
    return kind, _recv_exact(sock, n) if n else b""


def pcm_from(audio):
    """float32 [-1,1] → S16LE 바이트."""
    x = np.clip(np.asarray(audio, dtype=np.float32), -1.0, 1.0)
    return (x * 32767.0).astype("<i2").tobytes()


def pcm_to(data):
    """S16LE 바이트 → float32 [-1,1]."""
    return np.frombuffer(data, dtype="<i2").astype(np.float32) / 32768.0


def connect(host, port, timeout=10.0):
    s = socket.create_connection((host, port), timeout=timeout)
    s.settimeout(None)
    s.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)   # 조각을 모으지 말 것
    return s
