"""ROS 노드의 재생 상태 처리를 ROS 없이 시험한다.

여기서 보는 것은 두 가지다. 둘 다 리뷰에서 지적된 것이고, 하드웨어 없이
재현된다.

  ② 수락 응답이 오기 전에 취소가 들어오면 그 취소가 사라진다
  ③ 마지막 오디오 프레임과 EOS 가 겹쳐 나가 순서가 뒤집힐 수 있다

rclpy 를 깔지 않은 곳에서도 돌아야 하므로 모듈을 가짜로 채운 뒤 노드
파일에서 함수와 메서드만 꺼내 쓴다. Node.__init__ 은 부르지 않고 객체만
만들어, 시험할 메서드가 실제로 건드리는 속성만 채운다 — 진짜 메서드를
시험하기 위해서다.
"""
import os
import queue
import sys
import types

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)

for name in ("rclpy", "rclpy.node", "rclpy.qos", "rclpy.action",
             "rclpy.callback_groups", "rclpy.executors",
             "lamp_interfaces", "lamp_interfaces.msg",
             "lamp_interfaces.action", "builtin_interfaces",
             "builtin_interfaces.msg"):
    sys.modules.setdefault(name, types.ModuleType(name))
for mod, attrs in (("rclpy.node", ["Node"]),
                   ("rclpy.qos", ["HistoryPolicy", "QoSPresetProfiles",
                                  "QoSProfile", "ReliabilityPolicy"]),
                   ("rclpy.action", ["ActionClient"]),
                   ("rclpy.callback_groups", ["ReentrantCallbackGroup"]),
                   ("rclpy.executors", ["MultiThreadedExecutor"]),
                   ("lamp_interfaces.msg", ["AudioFrame", "OrientationStatus",
                                            "AudioStatus"]),
                   ("lamp_interfaces.action", ["PlayAudio"]),
                   ("builtin_interfaces.msg", ["Time"])):
    for a in attrs:
        setattr(sys.modules[mod], a, type(a, (), {}))

sys.path.insert(0, os.path.join(ROOT, "ros"))
import lamp_voice_node as N  # noqa: E402

DEFAULT_PREV_WAIT = N.PREV_PLAY_WAIT_S   # 시험들이 줄이기 전에 받아 둔다

fail = 0


def check(name, ok, detail=""):
    global fail
    print(f"  {'✔' if ok else '✗'} {name}{'  ' + detail if detail else ''}")
    if not ok:
        fail += 1


class Log:
    def info(self, *a): pass
    def warn(self, *a): pass
    def error(self, *a): pass


class Handle:
    """액션 목표 핸들 흉내. 취소 횟수를 세고 결과 콜백을 붙잡아 둔다."""

    def __init__(self, accepted=True):
        self.accepted = accepted
        self.cancels = 0
        self.result_cb = None

    def cancel_goal_async(self):
        self.cancels += 1

    def get_result_async(self):
        h = self

        class F:
            def add_done_callback(self, cb):
                h.result_cb = cb
        return F()

    def finish(self, code="drained", success=None):
        """액션 서버가 결과를 돌려준다."""
        r = types.SimpleNamespace(success=(code == "drained") if success is None
                                  else success, code=code, message="")
        self.result_cb(types.SimpleNamespace(
            result=lambda: types.SimpleNamespace(result=r)))


class Fut:
    def __init__(self, handle):
        self._h = handle

    def result(self):
        return self._h


N.PlayAudio.Goal = lambda **k: types.SimpleNamespace(**k)


class Client:
    """액션 클라이언트 흉내. 보낸 목표의 수락 콜백을 붙잡아 둔다."""

    def __init__(self):
        self.pending = []           # (목표, 수락 콜백)
        self.server = True

    def wait_for_server(self, timeout_sec=None):
        return self.server

    def send_goal_async(self, goal):
        c = self

        class F:
            def add_done_callback(self, cb):
                c.pending.append((goal, cb))
        return F()

    def accept(self, accepted=True):
        """가장 최근 목표를 수락(또는 거부)한다. 그 핸들을 돌려준다."""
        _, cb = self.pending[-1]
        h = Handle(accepted)
        cb(Fut(h))
        return h


def node():
    """__init__ 없이 만든다(rclpy 가 없다). 재생 경로가 쓰는 것만 채운다."""
    import threading
    n = object.__new__(N.LampVoiceNode)
    n.get_logger = lambda: Log()
    n.cur = None
    n.play_done = threading.Event()
    n.play_done.set()
    n.play_lock = threading.Lock()
    n.play_audio = Client()
    n.published = []
    n.playback = types.SimpleNamespace(
        publish=lambda m: n.published.append((time.time(), m)))
    n._frame = lambda data, eos, sid, seq: types.SimpleNamespace(
        data=data, end_of_stream=eos, stream_id=sid, sequence=seq)
    n.done = []                     # 판단부에 보낸 SPEAK_DONE (발화 id, code)

    def send(kind, payload=b""):
        if kind == N.SPEAK_DONE:
            n.done.append((payload[:36].decode().strip(),
                           payload[36:].decode()))
    n.send = send
    return n


def begin(n, sid):
    n.on_agent_message(N.SPEAK_BEGIN, N.pack_id(sid))


import threading  # noqa: E402
import time  # noqa: E402

N.SENDER_READY_S = 0.0              # 시험에서는 송신기 준비를 기다리지 않는다


# ② 수락 응답 전에 들어온 취소 ─────────────────────────────────────────
print("② 수락 전 취소")

n = node()
begin(n, "a")
n.cancel_playback()                 # 수락 응답이 오기 전에 끼어듦
h = n.play_audio.accept()           # 뒤늦게 수락됨
check("수락되자마자 취소를 보낸다", h.cancels == 1, f"cancel {h.cancels}회")
check("그 전에는 끝났다고 보지 않는다", not n.play_done.is_set())
h.finish("cancelled")
check("취소 결과 뒤 재생 완료가 선다", n.play_done.is_set())
check("판단부에 자기 id 로 알린다", n.done == [("a", "cancelled")], f"{n.done}")

n = node()
begin(n, "a")
h = n.play_audio.accept()
check("취소가 없으면 취소를 안 부른다", h.cancels == 0)
check("수락되면 발행을 시작한다", n.cur.accepted.is_set())


# ④ 0초 A / 1초 A 취소 / 4초 B / 5초 A 결과 (리뷰 시나리오) ────────────
print("\n④ 앞 재생의 늦은 결과")

N.PREV_PLAY_WAIT_S = 0.2
n = node()
begin(n, "A")
hA = n.play_audio.accept()
n.cancel_playback()                 # A 에 끼어듦 — 결과가 늦는다
begin(n, "B")                       # 상한이 지나 B 시작
hB = n.play_audio.accept()
hA.finish("cancelled")              # 이제야 A 의 결과
check("B 가 끝난 것으로 처리되지 않는다", not n.play_done.is_set())
check("A 의 결과는 A 의 id 로만 간다", n.done == [("A", "cancelled")], f"{n.done}")
check("B 는 아직 살아 있다", hB.cancels == 0 and n.cur.accepted.is_set())
hB.finish("drained")
check("B 의 결과는 B 로 처리된다", n.play_done.is_set())
check("B 의 완료가 B 의 id 로 간다", n.done[-1] == ("B", "drained"), f"{n.done}")

# A 가 B 시작 뒤에야 수락되면 A 만 취소하고 B 는 안 건드린다
n = node()
begin(n, "A")                       # A 수락 응답이 안 온다
begin(n, "B")                       # 상한이 지나 B 시작
_, acceptA = n.play_audio.pending[0]
hA = Handle(True)
acceptA(Fut(hA))                    # A 가 뒤늦게 수락됨
check("뒤늦게 수락된 A 를 취소한다", hA.cancels == 1, f"cancel {hA.cancels}회")
check("B 를 건드리지 않는다", n.cur.speech_id == "B" and n.cur.handle is None)


# ⑤ 턴이 바뀌는 사이 ───────────────────────────────────────────────────
print("\n⑤ 턴이 바뀌는 사이")

# ⑤-a A 의 결과가 B 가 A 를 기다리는 사이(상한 안)에 온다.
#     전에는 SPEAK_BEGIN 을 받자마자 발화 id 를 노드 필드에 적어서, 이때
#     오는 A 의 완료가 B 의 이름으로 나갔다 → B 가 시작도 전에 끝났다.
N.PREV_PLAY_WAIT_S = 2.0
n = node()
begin(n, "A")
hA = n.play_audio.accept()
n.cancel_playback()
t = threading.Thread(target=begin, args=(n, "B"))
t.start()                           # B 는 A 가 끝나기를 기다리며 막힌다
time.sleep(0.2)
goals_before_A_done = len(n.play_audio.pending)
hA.finish("cancelled")              # 그 사이에 A 의 결과
t.join(3.0)
check("A 의 결과는 A 의 이름으로 간다", n.done == [("A", "cancelled")], f"{n.done}")
check("A 가 끝나기 전에는 B 의 목표를 안 보낸다(보내면 audio_busy)",
      goals_before_A_done == 1, f"A 결과 전 목표 {goals_before_A_done}개")
check("B 는 새로 시작한다", n.cur.speech_id == "B" and not n.play_done.is_set())

# ⑤-b 상한이 지나 B 로 넘어간 뒤 B 가 수락 전에 끼어들린다.
#     전에는 goal_handle 이 A 것 그대로라 A 를 또 취소하고 B 는 살아남았다.
N.PREV_PLAY_WAIT_S = 0.2
n = node()
begin(n, "A")
hA = n.play_audio.accept()
n.cancel_playback()                 # A 결과가 안 온다
begin(n, "B")
a_before = hA.cancels
n.cancel_playback()                 # B 수락 전에 끼어듦
hB = n.play_audio.accept()
check("B 가 수락되자 취소된다", hB.cancels == 1, f"B cancel {hB.cancels}회")
check("B 의 취소가 A 로 새지 않는다", hA.cancels == a_before,
      f"A cancel {a_before} → {hA.cancels}회")

# ⑤-c 거부도 끝이다
n = node()
begin(n, "C")
n.play_audio.accept(accepted=False)
check("거부되면 재생 완료가 선다", n.play_done.is_set())
check("거부되면 판단부에 알린다", n.done == [("C", "rejected")], f"{n.done}")

# ⑤-d 액션 서버가 없어도 끝이다
n = node()
n.play_audio.server = False
begin(n, "D")
check("서버가 없으면 재생 완료가 선다", n.play_done.is_set())
check("서버가 없으면 판단부에 알린다", n.done == [("D", "no_server")], f"{n.done}")

# ⑤-e 결과는 한 번만 알린다
n = node()
begin(n, "E")
n.play_audio.accept(accepted=False)
n.cancel_playback()
check("거부 뒤 끼어들어도 한 번만 알린다", len(n.done) == 1, f"{n.done}")
# 안전망: 지금은 부르는 쪽이 먼저 막지만, _report 자체도 한 번만 알려야
# 한다. 부르는 쪽을 고치면 여기가 마지막 방어선이다.
n._report(n.cur, "again")
check("_report 는 두 번 불러도 한 번만 알린다", len(n.done) == 1, f"{n.done}")

# ⑤-f 안전망: 지난 스트림이 수락되면 cancel_pending 이 없어도 취소한다.
#     지금은 _abandon 이 cancel_pending 을 세워서 이 경로를 막지만, _abandon
#     을 고치면 여기가 마지막 방어선이다.
n = node()
begin(n, "A")
begin(n, "B")
_, acceptA = n.play_audio.pending[0]
hA = Handle(True)
import gc  # noqa: E402
A = next(o for o in gc.get_objects()
         if isinstance(o, N.Stream) and o.speech_id == "A")
A.cancel_pending = False                    # 앞 단계의 방어를 일부러 걷는다
acceptA(Fut(hA))
check("지난 스트림은 cancel_pending 없이도 취소된다", hA.cancels == 1,
      f"cancel {hA.cancels}회")


# ③ 마지막 오디오 프레임과 EOS ──────────────────────────────────────────
print("\n③ 마지막 프레임과 EOS 순서")

N.PREV_PLAY_WAIT_S = 3.0
n = node()
begin(n, "F")
n.play_audio.accept()
for _ in range(5):
    n.on_agent_message(N.SPEAK_AUDIO, b"\x00" * N.FRAME_BYTES)
n.on_agent_message(N.SPEAK_END, b"")
n.cur.thread.join(3.0)

seqs = [m.sequence for _, m in n.published]
eos = [m.end_of_stream for _, m in n.published]
gaps = [round(b[0] - a[0], 4) for a, b in zip(n.published, n.published[1:])]
check("순번이 0부터 1씩 는다", seqs == list(range(len(seqs))), f"{seqs}")
check("EOS 는 마지막 하나뿐", eos == [False] * 5 + [True])
# 20ms 를 딱 맞춰 요구하지 않는다. 절대 일정이라 한 프레임이 늦으면 다음
# 간격이 그만큼 짧아진다 — 드리프트를 안 쌓는 대가다. 막으려는 것은 그런
# 1~2ms 흔들림이 아니라 "여러 프레임을 마이크로초 안에 쏟는 것" 이다.
# 고치기 전 EOS 간격은 0에 가까웠다.
FLOOR = N.FRAME_INTERVAL_S / 2
check("마지막 프레임과 EOS 사이가 붙어 있지 않다", bool(gaps) and gaps[-1] >= FLOOR,
      f"{gaps[-1]*1000:.1f}ms" if gaps else "발행 없음")
check("몰아 보낸 구간이 없다", bool(gaps) and all(g >= FLOOR for g in gaps),
      f"최소 {min(gaps)*1000:.1f}ms" if gaps else "발행 없음")

# 발행 중에 취소되면 그 뒤로는 아무것도 안 보낸다. 발행 스레드가 루프
# 안에 들어간 뒤에 취소해야 한다 — 그 전에 취소하면 루프 앞의 검사에
# 걸려서, 루프 안의 검사가 빠져도 시험이 통과해 버린다(실제로 그랬다).
n = node()
begin(n, "G")
n.play_audio.accept()
n.on_agent_message(N.SPEAK_AUDIO, b"\x00" * N.FRAME_BYTES)
for _ in range(100):
    if n.published:
        break
    time.sleep(0.01)
n.cancel_playback()
for _ in range(3):
    n.on_agent_message(N.SPEAK_AUDIO, b"\x00" * N.FRAME_BYTES)
n.on_agent_message(N.SPEAK_END, b"")
n.cur.thread.join(2.0)
check("발행 중 취소 뒤로는 아무것도 안 보낸다", len(n.published) == 1,
      f"{len(n.published)}개 (취소 전 1개여야 한다)")

# 파이가 재생 도중 실패로 끝내면 그 스트림 발행을 멈춘다
n = node()
begin(n, "H")
h = n.play_audio.accept()
n.on_agent_message(N.SPEAK_AUDIO, b"\x00" * N.FRAME_BYTES)
for _ in range(100):
    if n.published:
        break
    time.sleep(0.01)
h.finish("failed")                  # 도중에 실패
for _ in range(3):
    n.on_agent_message(N.SPEAK_AUDIO, b"\x00" * N.FRAME_BYTES)
n.cur.thread.join(2.0)
check("서버가 끝낸 뒤로는 발행하지 않는다", len(n.published) == 1,
      f"{len(n.published)}개 (실패 전 1개여야 한다)")

# 브리지 계약: 앞 재생의 정지 제한(audio.play.stop timeout=10)보다 길게
# 기다려야 한다. 짧으면 취소가 느릴 때 다음 턴이 audio_busy 로 죽는다.
print("\n⑥ 브리지 계약")
check("앞 재생 대기가 브리지 정지 제한 10초보다 길다", DEFAULT_PREV_WAIT > 10.0,
      f"{DEFAULT_PREV_WAIT:.0f}초")

print("\n  ✔ 노드 상태 처리 이상 없음" if not fail else f"\n  ✗ {fail}건 실패")
sys.exit(1 if fail else 0)
