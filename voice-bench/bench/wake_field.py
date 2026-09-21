"""실제 마이크로 웨이크워드를 잰다.

지금까지의 숫자는 전부 노트북 녹음이다. 실제로 램프가 듣는 소리는 XVF3800 의
빔포밍·잡음제거·AGC 를 거쳐 랜을 타고 온 것이라 성격이 다르다. 좋아질지
나빠질지 모르므로 재야 한다.

    # 젯슨에서. 판단부(voice_agent) 는 꺼둔다 — 같은 포트를 쓴다.
    venvs/melo-onnx/bin/python bench/wake_field.py

    # 다른 창에서 ROS 노드
    source /opt/ros/jazzy/setup.bash
    source ~/talking-lamp-integration/jetson_ws/install/setup.bash
    python3 ros/lamp_voice_node.py --agent 127.0.0.1:5150

STT·TTS·LLM 을 올리지 않는다. 웨이크워드만 본다 — 2초면 뜨고 GPU 도 안 쓴다.
램프가 대답하지 않으므로 대화 중 헛깨움을 재는 동안 말을 끊지 않는다.

**점수를 전부 남긴다.** 그러면 임계값과 연속 창 수를 바꿔 가며 다시 부를
필요가 없다. 한 번 재고 표 전체를 뽑는다.
"""
import argparse
import json
import os
import socket
import sys
import threading
import time
import wave

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, ROOT)

from voice import link                                  # noqa: E402
from voice.wake import PHRASE, load_wake                 # noqa: E402

THS = (0.3, 0.5, 0.7, 0.9)
NEEDS = (1, 2, 3)
STEP_S = 0.08          # 점수 하나 = 80 ms


class Field:
    """마이크 프레임을 받아 점수만 남긴다. 아무것도 되돌려 보내지 않는다."""

    def __init__(self, model_path, threshold, need, keep_audio=False):
        self.rec = []          # (t, 점수, 말하는중)
        self.link_down = False
        # 재는 동안의 소리를 남긴다. 실제 채널을 거친 부정 데이터는 이렇게
        # 밖에 못 얻는데, 처음엔 점수만 남기고 소리를 버렸다. 5분을 다시
        # 받아야 했다.
        self.keep_audio = keep_audio
        self.audio = []        # (t, float32 320샘플)
        self.marks = []        # (구간이름, 시작, 끝)
        self.frames = 0
        self.active_now = False
        self.wake = load_wake(model_path, threshold, need,
                              on_score=self._score)
        if not getattr(self.wake, "ready", False):
            raise SystemExit(
                f"! 웨이크워드 모델이 없다({model_path}).\n"
                "  대역(아무 소리에나 깨어남)으로는 재봐야 의미가 없다.\n"
                "  git pull 로 models/wake/pixs-ya.onnx 를 받고,\n"
                "  venv 에 openwakeword 가 있는지 확인할 것:\n"
                "    venvs/melo-onnx/bin/pip install --no-deps openwakeword")

    def _score(self, s):
        self.rec.append((time.time(), float(s), self.active_now))

    def on_capture(self, speech_id, pcm):
        self.frames += 1
        self.active_now = bool(speech_id)
        if self.keep_audio:
            self.audio.append((time.time(), pcm.copy()))
        # 판단부와 같은 규칙: 파이가 말이라고 표시한 프레임만 넣는다.
        if self.active_now:
            self.wake.detect(pcm)

    def on_orientation(self, speech_id, state):
        pass

    def mark(self, label, t0, t1):
        self.marks.append((label, t0, t1))


def fired_runs(scores, th, need):
    """임계를 need 번 연속 넘은 횟수. 연속으로 넘는 동안은 한 번으로 센다."""
    n, run, armed = 0, 0, True
    for s in scores:
        if s < th:
            run, armed = 0, True
            continue
        run += 1
        if run >= need and armed:
            n += 1
            armed = False
    return n


def window(rec, t0, t1):
    return [s for t, s, _ in rec if t0 <= t <= t1]


def report(rec, marks, out_path):
    calls = [m for m in marks if m[0] == "call" and window(rec, m[1], m[2])]
    skipped = sum(1 for m in marks
                  if m[0] == "call" and not window(rec, m[1], m[2]))
    talks = [m for m in marks if m[0] == "talk"]
    talk_s = sum(t1 - t0 for _, t0, t1 in talks)

    print(f"\n{'='*58}")
    print(f"부른 횟수 {len(calls)}회, 대화 {talk_s/60:.1f}분, 점수 {len(rec)}개")
    if calls:
        mx = sorted(max(window(rec, t0, t1)) for _, t0, t1 in calls)
        # 못 깨어난 것이 아깝게 못 넘긴 건지 아예 0점인지 갈라야 한다.
        # 앞이면 임계값 문제고, 뒤면 모델이 그 소리를 아예 모른다는 뜻이라
        # 임계값을 아무리 내려도 안 된다.
        near = sum(1 for v in mx if 0.1 <= v < 0.7)
        zero = sum(1 for v in mx if v < 0.1)
        print(f"\n부를 때마다의 최고 점수 (낮은 순)")
        print("  " + "  ".join(f"{v:.2f}" for v in mx))
        print(f"  0.7 이상 {len(mx)-near-zero}회 / "
              f"아깝게 못 넘김(0.1~0.7) {near}회 / "
              f"거의 0점 {zero}회")
        if zero > near:
            print("  → 임계값 문제가 아니다. 모델이 그 소리를 아예 못 알아본다")
        elif near:
            print("  → 임계값을 내리면 일부는 건진다")
    if skipped:
        print(f"! {skipped}회는 파이 VAD 가 말로 보지 않아 뺐다 — 마이크 쪽 문제다")
    print(f"{'='*58}\n")
    print(f"{'연속':>4} {'임계':>5} {'깨어남':>12} {'대화 중 헛깨움':>16}")
    print("-" * 42)
    best = None
    for need in NEEDS:
        for th in THS:
            hit = sum(1 for _, t0, t1 in calls
                      if fired_runs(window(rec, t0, t1), th, need) > 0)
            fa = sum(fired_runs(window(rec, t0, t1), th, need)
                     for _, t0, t1 in talks)
            rate = 100.0 * hit / len(calls) if calls else 0.0
            per_min = fa / (talk_s / 60) if talk_s else 0.0
            print(f"{need:>4} {th:>5.1f} {hit:>4}/{len(calls):<3} {rate:>4.0f}%"
                  f" {fa:>6}회 {per_min:>6.2f}회/분")
            # 헛깨움이 같으면 더 보수적인 자리(연속 많고 임계 높은)를 고른다
            key = (per_min, -need, -th)
            if rate >= 95 and (best is None or key < best[0]):
                best = (key, need, th, rate, per_min)
        print()
    if best:
        _, need, th, rate, per_min = best
        print(f"깨어남 95% 이상 중 헛깨움이 가장 적은 자리: 연속 {need}창, 임계 {th}")
        print(f"  깨어남 {rate:.0f}%, 헛깨움 {per_min:.2f}회/분")
    else:
        print("깨어남 95% 를 넘는 자리가 없다. 실제 마이크에서 잘 안 들린다는 뜻이다.")

    with open(out_path, "w") as f:
        json.dump({"scores": [[round(t, 3), round(s, 4), a] for t, s, a in rec],
                   "marks": [[m, round(a, 3), round(b, 3)] for m, a, b in marks],
                   "step_s": STEP_S}, f)
    print(f"\n원자료 {os.path.relpath(out_path, ROOT)}")
    print("점수를 다 남겼으므로 다른 임계값이 궁금하면 다시 부를 필요 없다.")


def save_audio(f, marks, out_dir, ths=(0.7,), need=2):
    """대화 구간 소리를 통째로 저장하고, 헛깨운 자리는 따로 잘라 둔다.

    헛깨운 2초는 그냥 부정이 아니라 모델이 실제로 속은 소리다. 학습에서
    가장 값이 나가는 쪽이라 찾기 쉽게 따로 뺀다.
    """
    if not f.audio:
        return
    os.makedirs(out_dir, exist_ok=True)
    os.makedirs(os.path.join(out_dir, "속은것"), exist_ok=True)
    stamp = time.strftime("%Y%m%d-%H%M%S")

    def write(path, chunks):
        pcm = np.concatenate(chunks) if chunks else np.zeros(1, np.float32)
        with wave.open(path, "wb") as w:
            w.setnchannels(1); w.setsampwidth(2); w.setframerate(16000)
            w.writeframes((np.clip(pcm, -1, 1) * 32767).astype("<i2").tobytes())

    talks = [m for m in marks if m[0] == "talk"]
    n_clip = 0
    for i, (_, t0, t1) in enumerate(talks):
        chunks = [p for t, p in f.audio if t0 <= t <= t1]
        if not chunks:
            continue
        write(os.path.join(out_dir, f"대화-{stamp}-{i}.wav"), chunks)

        # 헛깨운 순간 앞 2초
        scores = [(t, sc) for t, sc, _ in f.rec if t0 <= t <= t1]
        run, armed = 0, True
        for t, sc in scores:
            if sc < ths[0]:
                run, armed = 0, True
                continue
            run += 1
            if run >= need and armed:
                armed = False
                clip = [p for tt, p in f.audio if t - 2.0 <= tt <= t]
                if clip:
                    write(os.path.join(out_dir, "속은것",
                                       f"{stamp}-{n_clip:03d}.wav"), clip)
                    n_clip += 1
    print(f"  소리 저장 {os.path.relpath(out_dir, ROOT)} "
          f"(속은 것 {n_clip}개 따로 뺐다)")


def pump(sock, f):
    """노드가 끊기면 표시만 하고 조용히 끝낸다.

    그냥 죽게 두면 프레임이 안 오는 채로 계속 물어보게 되고, 결과가
    "깨어남 0%" 로 나와 모델 탓처럼 보인다. 연결이 끊긴 것과 모델이 못
    알아듣는 것은 완전히 다른 얘기다.
    """
    try:
        _pump(sock, f)
    except (ConnectionError, OSError) as e:
        f.link_down = True
        print(f"\n  ! ROS 노드 연결이 끊겼다: {e}")


def _pump(sock, f):
    while True:
        kind, body = link.recv(sock)
        if kind == link.CAP_FRAME:
            sid, pcm = link.unpack_id(body)
            f.on_capture(sid, link.pcm_to(pcm))
        elif kind == link.ORIENT:
            sid, st = link.unpack_id(body)
            f.on_orientation(sid, st.decode("utf-8", "replace"))
        elif kind == link.PING:
            link.send(sock, link.PONG)


def load_runs(paths):
    """따로 잰 결과들을 하나로 합친다.

    부르기와 대화를 다른 날 재도 된다. 구간마다 자기 시각을 들고 있으므로
    그냥 이어 붙이면 섞이지 않는다.
    """
    rec, marks = [], []
    for path in paths:
        with open(path) as f:
            d = json.load(f)
        rec += [(t, s, a) for t, s, a in d["scores"]]
        marks += [(m, a, b) for m, a, b in d["marks"]]
    rec.sort(key=lambda r: r[0])
    return rec, marks


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--report", nargs="+", metavar="JSON",
                    help="이미 잰 결과 파일들로 표만 다시 뽑는다. "
                         "여러 개를 주면 합쳐서 센다")
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--port", type=int, default=5150)
    ap.add_argument("--wake-model", default="models/wake/pixs-ya.onnx")
    ap.add_argument("--calls", type=int, default=20, help="몇 번 불러볼지")
    ap.add_argument("--call-window", type=float, default=4.0,
                    help="한 번 부를 때 기다리는 시간(초)")
    ap.add_argument("--talk-min", type=float, default=5.0,
                    help="헛깨움을 잴 대화 시간(분)")
    ap.add_argument("--out", default=os.path.join(ROOT, "out"))
    ap.add_argument("--save-audio", metavar="DIR", nargs="?",
                    const=os.path.join(ROOT, "wake-data-real"),
                    help="대화 구간 소리를 남긴다. 실제 채널을 거친 부정 "
                         "데이터는 이렇게밖에 못 얻는다")
    ap.add_argument("--check", action="store_true",
                    help="모델이 열리는지만 보고 끝낸다. 재기 전에 먼저 할 것")
    a = ap.parse_args()

    if a.report:
        rec, marks = load_runs(a.report)
        print(f"결과 {len(a.report)}개를 합쳐서 센다")
        report(rec, marks, os.path.join(a.out, "wake-field-합본.json"))
        return

    f = Field(os.path.join(ROOT, a.wake_model), 0.5, 1,   # 기록만 — 판정은 나중에
              keep_audio=bool(a.save_audio))
    if a.check:
        import numpy as np
        for i in range(60):        # 1.2초치 무음을 넣어 실제로 돌려 본다
            f.active_now = True
            f.wake.detect(np.zeros(320, dtype=np.float32))
        print(f"  {f.wake.name}")
        print(f"  점수 {len(f.rec)}개 나옴 — 모델이 정상으로 돈다")
        print("\n준비됐다. --check 없이 다시 띄우고 ROS 노드를 붙일 것")
        return
    print(f"  {f.wake.name}")
    print("  (여기서는 판정하지 않고 점수만 남긴다. 표는 끝나고 뽑는다)")

    srv = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    srv.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    srv.bind((a.host, a.port))
    srv.listen(1)
    print(f"\n  ROS 노드를 기다린다 {a.host}:{a.port}")
    sock, addr = srv.accept()
    sock.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
    print(f"  붙었다 {addr[0]}:{addr[1]}")
    threading.Thread(target=pump, args=(sock, f), daemon=True).start()

    time.sleep(1.0)
    if f.frames == 0:
        print("\n  ! 마이크 프레임이 안 온다. 파이 device 서비스와 브리지를 확인할 것")
    else:
        print(f"  마이크 프레임 들어오는 중 ({f.frames}개)")

    if a.calls == 0:
        print("\n① 부르기 — 건너뛴다(--calls 0)")
    else:
        print(f"\n{'='*58}")
        print(f'① 부르기 — "{PHRASE}" 를 {a.calls}번 부른다')
        print(f"{'='*58}")
        print("매번 Enter 를 누르고 부른다. 말투와 거리를 바꿔 가며 부를 것.")
        print("(평소, 작게, 크게, 빠르게, 천천히, 멀리서, 고개 돌린 채)")
    for i in range(a.calls):
        input(f"\n  [{i+1}/{a.calls}] Enter 누르고 부르세요 → ")
        t0 = time.time()
        time.sleep(a.call_window)
        t1 = time.time()
        f.mark("call", t0, t1)
        got_all = window(f.rec, t0, t1)
        if f.link_down:
            raise SystemExit("  ROS 노드가 끊겼다. 다시 띄우고 처음부터 할 것")
        if not got_all:
            # 파이 VAD 가 말이라고 표시하지 않았다는 뜻이다. 모델까지
            # 가보지도 못했으므로 모델 탓이 아니다.
            print("      ! 점수가 하나도 없다 — 파이 VAD 가 말로 보지 않았다")
            continue
        got = max(got_all)
        print(f"      최고 점수 {got:.3f}" + ("  ✔" if got >= 0.7 else "  ✗"))

    if a.talk_min <= 0:
        print("\n② 헛깨움 — 건너뛴다(--talk-min 0). 나중에 이렇게 돌린다:")
        print("     venvs/melo-onnx/bin/python bench/wake_field.py --calls 0")
    else:
        print(f"\n{'='*58}")
        print(f"② 헛깨움 — {a.talk_min:.0f}분 동안 평소처럼 대화한다")
        print(f"{'='*58}")
        print(f'호출어("{PHRASE}") 는 말하지 말 것. 여러 명이면 더 좋다.')
        print("램프는 대답하지 않는다 — 점수만 센다.")
        # input() 까지 try 안에 넣는다. 밖에 두면 여기서 Ctrl+C 를 누를 때
        # 프로그램이 그냥 죽어서 앞서 부른 20번이 통째로 날아간다.
        t0 = time.time()
        try:
            input("\n  준비되면 Enter (지금 Ctrl+C 해도 ①은 저장된다) → ")
            t0 = time.time()
            end = t0 + a.talk_min * 60
            while time.time() < end:
                left = end - time.time()
                n = fired_runs(window(f.rec, t0, time.time()), 0.7, 2)
                print(f"\r  남은 시간 {int(left)//60}:{int(left)%60:02d}   "
                      f"지금까지 헛깨움 {n}회 (임계 0.7 기준)", end="", flush=True)
                time.sleep(1.0)
        except KeyboardInterrupt:
            print("\n  중단 — 여기까지로 센다")
        if time.time() - t0 > 5:
            f.mark("talk", t0, time.time())
        print()

    os.makedirs(a.out, exist_ok=True)
    stamp = time.strftime("%Y%m%d-%H%M%S")
    if a.save_audio:
        save_audio(f, f.marks, a.save_audio)
    report(f.rec, f.marks, os.path.join(a.out, f"wake-field-{stamp}.json"))


if __name__ == "__main__":
    main()
