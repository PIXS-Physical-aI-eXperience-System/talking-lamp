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
from lamp_voice_node import SPEAK_DONE  # noqa: E402

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
    """액션 목표 핸들 흉내. 취소가 불렸는지 센다."""

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


class Fut:
    def __init__(self, handle):
        self._h = handle

    def result(self):
        return self._h


def bare_node():
    """__init__ 없이 만들고 시험할 메서드가 쓰는 속성만 채운다."""
    n = object.__new__(N.LampVoiceNode)
    n.get_logger = lambda: Log()
    n.goal_handle = None
    n.cancel_pending = False
    n.goal_sent = False
    n.cancelled = False
    n.sent = 0
    n.stream_id = "s1"
    n.speech_id = "u1"
    n.outq = queue.Queue()
    import threading
    n.play_done = threading.Event()
    n.play_done.clear()
    n.accepted = threading.Event()
    n.sent_to_agent = []
    n.done_payloads = []

    def _send(kind, payload=b""):
        n.sent_to_agent.append(kind)
        if kind == N.SPEAK_DONE:
            n.done_payloads.append((kind, payload))
    n.send = _send
    n.play_t0 = 0.0
    return n


# ② 수락 전에 들어온 취소가 사라지지 않는다 ────────────────────────────
print("② 수락 전 취소")

n = bare_node()
n.goal_sent = True                      # 목표는 보냈고 응답을 기다리는 중
n.cancel_playback()                     # 여기서 barge-in
check("수락 전 취소는 보류된다", n.cancel_pending is True)
check("발행 스레드에 끝 신호를 넣는다", n.outq.qsize() == 1)

h = Handle(accepted=True)
n._goal_accepted("s1", Fut(h))          # 뒤늦게 수락됨
check("수락되자마자 취소를 보낸다", h.cancels == 1, f"cancel {h.cancels}회")
check("보류 표시를 지운다", n.cancel_pending is False)
check("결과 콜백을 단다", h.result_cb is not None)

# 결과가 오면 상태가 정리된다
class R:
    success, code, message = False, "cancelled", ""


class RF:
    def result(self):
        return types.SimpleNamespace(result=R())


n._goal_result("s1", RF())
check("결과 뒤 goal_handle 이 비워진다", n.goal_handle is None)
check("결과 뒤 재생 완료가 선다", n.play_done.is_set())

# 취소가 없었으면 평소대로 동작한다
n2 = bare_node()
n2.goal_sent = True
h2 = Handle(accepted=True)
n2._goal_accepted("s1", Fut(h2))
check("취소가 없으면 취소를 안 부른다", h2.cancels == 0)
check("평소에는 handle 을 들고 있는다", n2.goal_handle is h2)

# ④ 앞 스트림의 늦은 결과가 지금 재생을 끝내면 안 된다 ────────────────
#    0초 A 시작 / 1초 A 취소 요청 / 4초 상한이 지나 B 시작 /
#    5초 A 의 취소 결과 도착 → 그때 B 가 끝난 것으로 처리되면 안 된다.
print("\n④ 앞 스트림의 늦은 결과")

n4 = bare_node()
n4.stream_id = "A"
hA = Handle(accepted=True)
n4._goal_accepted("A", Fut(hA))
check("A 가 수락된다", n4.goal_handle is hA)

# 상한이 지나 B 가 시작된 상태를 만든다
n4.stream_id = "B"
n4.play_done.clear()
n4.goal_handle = None
n4.sent_to_agent.clear()

n4._goal_result("A", RF())              # A 의 늦은 결과
check("B 가 끝난 것으로 처리되지 않는다", not n4.play_done.is_set())
check("판단부에 완료를 보내지 않는다", SPEAK_DONE not in n4.sent_to_agent,
      f"{[k.decode() for k in n4.sent_to_agent]}")

# 지금 스트림(B)의 결과는 정상으로 처리된다
n4._goal_result("B", RF())
check("B 의 결과는 처리된다", n4.play_done.is_set())
check("B 의 완료는 판단부로 간다", SPEAK_DONE in n4.sent_to_agent)
check("완료에 발화 id 가 실린다",
      any(p[:36].decode().strip() == "u1" for k, p in n4.done_payloads),
      f"{[p[:36].decode().strip() for k, p in n4.done_payloads]}")

# 앞 스트림이 뒤늦게 수락되면 그것만 취소하고 지금 것을 안 건드린다
n5 = bare_node()
n5.stream_id = "B"
n5.goal_handle = "B의핸들"
hLate = Handle(accepted=True)
n5._goal_accepted("A", Fut(hLate))
check("늦게 수락된 A 를 취소한다", hLate.cancels == 1, f"cancel {hLate.cancels}회")
check("B 의 핸들을 안 덮는다", n5.goal_handle == "B의핸들")

# ③ 마지막 오디오 프레임과 EOS 사이 간격 ──────────────────────────────
print("\n③ 마지막 프레임과 EOS 순서")

N.SENDER_READY_S = 0.0                  # 시험에서는 기다리지 않는다
import time  # noqa: E402

n3 = bare_node()
n3.play_done.set()
n3.accepted.set()                       # 목표가 수락된 상태
published = []


class Pub:
    def publish(self, msg):
        published.append((time.time(), msg))


n3.playback = Pub()
n3._frame = lambda data, eos, sid, seq: types.SimpleNamespace(
    data=data, end_of_stream=eos, stream_id=sid, sequence=seq)

for _ in range(5):
    n3.outq.put(b"\x00" * N.FRAME_BYTES)
n3.outq.put(None)                       # EOS
n3._publisher("s1", n3.outq)

seqs = [m.sequence for _, m in published]
eos = [m.end_of_stream for _, m in published]
gaps = [round(b[0] - a[0], 4) for a, b in zip(published, published[1:])]
check("순번이 0부터 1씩 는다", seqs == list(range(len(seqs))), f"{seqs}")
check("EOS 는 마지막 하나뿐", eos == [False] * 5 + [True])
# 20ms 를 딱 맞춰 요구하지 않는다. 절대 일정이라 한 프레임이 늦으면 다음
# 간격이 그만큼 짧아진다 — 드리프트를 안 쌓는 대가다. 막으려는 것은 그런
# 1~2ms 흔들림이 아니라 "여러 프레임을 마이크로초 안에 쏟는 것" 이다.
# 고치기 전 EOS 간격은 0에 가까웠다.
FLOOR = N.FRAME_INTERVAL_S / 2
check("마지막 프레임과 EOS 사이가 붙어 있지 않다", gaps[-1] >= FLOOR,
      f"{gaps[-1]*1000:.1f}ms (10ms 이상이어야 한다)")
check("몰아 보낸 구간이 없다", all(g >= FLOOR for g in gaps),
      f"최소 {min(gaps)*1000:.1f}ms")

print("\n  ✔ 노드 상태 처리 이상 없음" if not fail else f"\n  ✗ {fail}건 실패")
sys.exit(1 if fail else 0)
