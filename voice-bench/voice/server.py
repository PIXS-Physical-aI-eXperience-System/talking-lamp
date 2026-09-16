"""젯슨 쪽 — STT·TTS 를 들고 파이의 요청을 받는다.

모델은 한 번만 올린다. 적재에 40초 넘게 걸리므로 요청마다 올릴 수 없다.

한 번에 한 연결만 받는다. 램프는 하나고, 동시에 두 요청이 오면 GPU 메모리
최고점이 예산을 넘는다. 여러 개를 받아야 할 이유가 생기면 그때 늘린다.
"""
import socket
import time

from . import link


class VoiceServer:
    def __init__(self, stt, tts, host="0.0.0.0", port=5150, chunk_chars=60):
        self.stt = stt
        self.tts = tts
        self.host, self.port = host, port
        self.chunk_chars = chunk_chars

    def serve_forever(self):
        srv = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        srv.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        srv.bind((self.host, self.port))
        srv.listen(1)
        print(f"  대기 중 {self.host}:{self.port}")
        while True:
            sock, addr = srv.accept()
            sock.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
            print(f"  연결됨 {addr[0]}:{addr[1]}")
            try:
                self._handle(sock)
            except ConnectionError as e:
                print(f"  연결 끊김: {e}")
            except Exception as e:
                print(f"  오류: {type(e).__name__}: {e}")
            finally:
                sock.close()

    def _handle(self, sock):
        while True:
            kind, body = link.recv(sock)
            if kind == link.STT_REQ:
                t0 = time.perf_counter()
                audio = link.pcm_to(body)
                text = self.stt.transcribe(audio, link.RATE)
                print(f"  STT {len(audio)/link.RATE:.1f}s → "
                      f"{time.perf_counter()-t0:.2f}s  \"{text}\"")
                link.send(sock, link.STT_RES, text.encode("utf-8"))

            elif kind == link.TTS_REQ:
                text = body.decode("utf-8", "replace")
                t0 = time.perf_counter()
                first = None
                # 문장 단위로 만들어 만드는 대로 보낸다. 다 만들고 보내면
                # 파이에서 첫 소리가 나기까지 전부 기다려야 한다.
                from .tts import split_sentences
                for part in split_sentences(text, self.chunk_chars):
                    audio = self.tts.synth(part)
                    if first is None:
                        first = time.perf_counter() - t0
                    src = self.tts.samplerate
                    if src != link.RATE:
                        from .audio import resample
                        audio = resample(audio, src, link.RATE)
                    link.send(sock, link.TTS_AUD, link.pcm_from(audio))
                link.send(sock, link.TTS_END)
                print(f"  TTS {len(text)}자 → 첫 조각 {first:.2f}s, "
                      f"전체 {time.perf_counter()-t0:.2f}s")

            elif kind == link.PING:
                link.send(sock, link.PONG)
            else:
                link.send(sock, link.ERROR, b"unknown")
