"""판단부와 ROS 노드를 실제 코드 그대로 이어서 돌린다 — ROS·장치 없이.

agent_test 와 node_test 는 각자를 따로 본다. 둘 사이 규약(발화 id 를 싣고
돌려주는 것, 누가 언제 끝났다고 말하는지)이 맞물리는지는 이어야 보인다.

이음새만 가짜다.
  판단부 → 노드   큐 + 스레드 (실제로는 소켓 + read_loop)
  노드 → 판단부   큐 + 스레드 (실제로는 소켓 + voice_agent 의 pump)
  액션 서버       시험이 수락·결과를 손으로 준다

리뷰에서 나온 시나리오를 그대로 재생한다:
  0초 A 재생 / 1초 끼어들어 A 취소 / A 결과가 늦음 / B 시작 /
  A 의 결과 도착 → B 가 끝난 것으로 처리되면 안 된다
"""
import os
import queue
import sys
import threading
import time
import types

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, ROOT)
sys.path.insert(0, HERE)

# ── rclpy 없이 노드를 올린다 ──────────────────────────────────────────
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

from agent_test import SID, AlwaysWake, FakeStt, FakeTts, frame  # noqa: E402
from voice import link  # noqa: E402
from voice.agent import IDLE, LISTENING, SPEAKING, VoiceAgent  # noqa: E402

N.SENDER_READY_S = 0.0
N.PREV_PLAY_WAIT_S = 0.3
N.PlayAudio.Goal = lambda **k: types.SimpleNamespace(**k)

fail = 0


def check(name, ok, detail=""):
    global fail
    print(f"  {'✔' if ok else '✗'} {name}{'  ' + detail if detail else ''}")
    if not ok:
        fail += 1


def wait_for(pred, timeout=3.0):
    t = time.time() + timeout
    while time.time() < t:
        if pred():
            return True
        time.sleep(0.01)
    return False


class Handle:
    def __init__(self):
        self.accepted = True
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

    def finish(self, code):
        r = types.SimpleNamespace(success=code == "drained", code=code, message="")
        self.result_cb(types.SimpleNamespace(
            result=lambda: types.SimpleNamespace(result=r)))


class Server:
    """액션 서버 흉내. 목표를 붙잡아 두고 시험이 수락을 준다."""

    def __init__(self):
        self.goals = []             # (목표, 수락 콜백)

    def wait_for_server(self, timeout_sec=None):
        return True

    def send_goal_async(self, goal):
        s = self

        class F:
            def add_done_callback(self, cb):
                s.goals.append((goal, cb))
        return F()

    def accept(self, i):
        h = Handle()
        self.goals[i][1](types.SimpleNamespace(result=lambda: h))
        return h


# ── 잇기 ──────────────────────────────────────────────────────────────
to_node, to_agent = queue.Queue(), queue.Queue()
log = []                            # (방향, 종류, 시각)

node = object.__new__(N.LampVoiceNode)
node.get_logger = lambda: types.SimpleNamespace(
    info=lambda *a: None, warn=lambda *a: None, error=lambda *a: None)
node.cur = None
node.play_done = threading.Event()
node.play_done.set()
node.play_lock = threading.Lock()
node.play_audio = Server()
node.published = []
node.playback = types.SimpleNamespace(publish=node.published.append)
node._frame = lambda data, eos, sid, seq: types.SimpleNamespace(
    stream_id=sid, sequence=seq, end_of_stream=eos)
node.send = lambda kind, payload=b"": to_agent.put((kind, payload))


def agent_send(kind, payload=b""):
    log.append(("→노드", kind))
    to_node.put((kind, payload))


agent = VoiceAgent(FakeStt(), FakeTts(), AlwaysWake(),
                   on_utterance=lambda t: "네, 밝게 할게요.",
                   send=agent_send, on_state=lambda s: None)
agent.barge.hold_s = 0.0


def node_loop():                    # 노드의 read_loop
    while True:
        kind, body = to_node.get()
        node.on_agent_message(kind, body)


def agent_loop():                   # voice_agent 의 pump
    while True:
        kind, body = to_agent.get()
        log.append(("→판단부", kind))
        if kind == link.SPEAK_DONE:
            sid, code = link.unpack_id(body)
            agent.on_play_done(sid, code.decode())


threading.Thread(target=node_loop, daemon=True).start()
threading.Thread(target=agent_loop, daemon=True).start()


def feed(script):
    for sid, db, n in script:
        for _ in range(n):
            agent.on_capture(sid, frame(db))
            time.sleep(0.001)


# ── 턴 A ──────────────────────────────────────────────────────────────
print("턴 A — 말하는 중에 끼어든다")
feed([(SID, -20, 50), ("", -60, 35)])
check("A 가 목표를 보낸다", wait_for(lambda: len(node.play_audio.goals) == 1))
hA = node.play_audio.accept(0)
streamA = node.cur
check("A 의 끝 신호까지 판단부가 보낸다",
      wait_for(lambda: ("→노드", link.SPEAK_END) in log))
check("A 는 재생이 끝나기 전이라 말하기다", agent.state == SPEAKING, agent.state)

feed([("", -55, 20), ("", -27, 5)])            # 바닥 → 끼어듦
check("끼어들면 듣기로 간다", agent.state == LISTENING, agent.state)
check("노드가 A 를 취소한다", wait_for(lambda: hA.cancels == 1),
      f"cancel {hA.cancels}회")

# ── 턴 B — A 의 결과가 안 온 채로 ───────────────────────────────────────
print("\n턴 B — A 의 결과가 아직 안 왔다")
feed([("", -20, 20), ("", -60, 30)])          # 끼어든 말이 끝남
check("B 가 목표를 보낸다(상한이 지나서)",
      wait_for(lambda: len(node.play_audio.goals) == 2))
hB = node.play_audio.accept(1)
streamB = node.cur
check("노드의 지금 재생은 B 다", streamB is not streamA)
check("B 는 말하기다", wait_for(lambda: agent.state == SPEAKING), agent.state)
check("B 의 끝 신호까지 보낸다",
      wait_for(lambda: [k for d, k in log if d == "→노드"].count(link.SPEAK_END) == 2))

# ── 이제야 A 의 결과 ─────────────────────────────────────────────────
print("\nA 의 취소 결과가 늦게 도착")
hA.finish("cancelled")
time.sleep(0.3)
check("B 는 여전히 말하기다", agent.state == SPEAKING, agent.state)
check("노드는 B 가 끝났다고 보지 않는다", not node.play_done.is_set())
check("B 는 취소되지 않았다", hB.cancels == 0)

# ── B 의 결과 ─────────────────────────────────────────────────────────
# 파이는 EOS 를 받고 다 틀어야 drained 를 준다. 그 전에 결과를 주면
# 실제로는 일어날 수 없는 순서를 시험하게 된다(처음에 그렇게 짰다).
print("\nB 가 실제로 끝남")
check("B 의 EOS 가 나간다", wait_for(lambda: any(
    m.end_of_stream and m.stream_id == streamB.id for m in node.published)))
hB.finish("drained")
check("B 가 끝나면 대기로 간다", wait_for(lambda: agent.state == IDLE), agent.state)
check("노드도 끝났다고 본다", node.play_done.is_set())

# ── 스트림이 안 섞였는가 ─────────────────────────────────────────────
b_frames = [m for m in node.published if m.stream_id == streamB.id]
b_seq = [m.sequence for m in b_frames]
check("B 의 순번이 0부터 이어진다", b_seq == list(range(len(b_seq))) and b_seq,
      f"{len(b_seq)}개")
check("B 의 EOS 는 마지막 하나", [m.end_of_stream for m in b_frames][-1:] == [True]
      and sum(m.end_of_stream for m in b_frames) == 1)
done_to_agent = sum(1 for d, k in log if d == "→판단부" and k == link.SPEAK_DONE)
check("완료 신호는 스트림마다 한 번씩", done_to_agent == 2, f"{done_to_agent}회")

print("\n  ✔ 판단부·노드 이음 이상 없음" if not fail else f"\n  ✗ {fail}건 실패")
sys.exit(1 if fail else 0)
