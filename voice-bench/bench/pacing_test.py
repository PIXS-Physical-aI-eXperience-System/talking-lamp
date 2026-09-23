"""재생 일정이 몰아 보내지 않는지 시험한다 — ROS 없이.

실제로 겪은 것: TTS 가 다음 문장을 만드는 동안 큐가 비고, 그동안 일정이
밀린다. 밀린 만큼 따라잡으려고 몰아 보내면 브리지가 순서를 뒤집어
`out_of_order: audio sequence is not the next frame` 로 스트림이 통째로
죽는다. 소리가 안 난다.
"""
import os
import sys
import types

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)

# rclpy 가 없는 곳에서도 돌아야 한다. 노드 파일에서 순수 함수만 꺼내 쓴다.
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
from lamp_voice_node import FRAME_INTERVAL_S, MIN_GAP_S, next_due  # noqa: E402

I = FRAME_INTERVAL_S
fail = 0


def check(name, ok, detail=""):
    global fail
    print(f"  {'✔' if ok else '✗'} {name}{'  ' + detail if detail else ''}")
    if not ok:
        fail += 1


# ① 큐가 늘 차 있으면 정확히 20ms 간격
t0, sent = 0.0, []
now = 0.0
for seq in range(10):
    due, t0, behind = next_due(t0, seq, now)
    now = max(now, due) + 0.001      # 발행에 1ms 걸린다고 치자
    sent.append(due)
gaps = [round(b - a, 4) for a, b in zip(sent, sent[1:])]
check("큐가 찼을 때 간격 20ms", all(abs(g - I) < 1e-6 for g in gaps), f"{gaps[:3]}")

# ② 오차가 쌓이지 않는다 — 발행이 매번 늦어도 절대 시각을 지킨다
t0, now = 0.0, 0.0
for seq in range(100):
    due, t0, _ = next_due(t0, seq, now)
    now = max(now, due) + 0.003
check("100프레임 뒤 누적 오차", abs(due - 99 * I) < 1e-6,
      f"{(due - 99*I)*1000:.3f}ms")

# ③ **큐가 3초 비었다가 다시 차도 몰아 보내지 않는다**
t0, now = 0.0, 0.0
sent = []
for seq in range(5):                       # 정상 5프레임
    due, t0, _ = next_due(t0, seq, now)
    now = max(now, due) + 0.001
    sent.append(due)
now += 3.0                                 # TTS 기다림
for seq in range(5, 30):                   # 다시 프레임이 쏟아진다
    due, t0, _ = next_due(t0, seq, now)
    now = max(now, due) + 0.001
    sent.append(due)
gaps = [round(b - a, 4) for a, b in zip(sent[5:], sent[6:])]
burst = [g for g in gaps if g < I - 1e-6]
check("큐가 빈 뒤에도 몰아 보내지 않는다", not burst,
      f"20ms 미만 간격 {len(burst)}개")

# ④ 밀린 것을 알려준다
_, _, behind = next_due(0.0, 5, 3.1)
check("밀린 시간을 돌려준다", behind > 2.9, f"{behind:.2f}초")

# ⑤ 한 프레임 이내로 늦은 것은 다시 잡지 않는다(정상 지터)
due, nt0, behind = next_due(0.0, 5, 5 * I + 0.005)
check("작은 지터는 그냥 둔다", behind == 0.0 and nt0 == 0.0)

# ⑥ 앞 프레임이 거의 한 칸 늦어도 다음 것이 붙어서 나가지 않는다.
#    절대 일정만 쓰면 늦은 만큼 다음 간격을 줄여 따라잡는데, 19ms 늦으면
#    다음 간격이 1ms 가 된다. 시험 60번에 한 번 EOS 가 0.1ms 뒤에 나갔다.
t0, now, last = 0.0, 0.0, None
sent = []
for seq in range(12):
    due, t0, _ = next_due(t0, seq, now, last=last)
    now = max(now, due)
    if seq == 5:
        now += 0.019                     # 이 프레임만 OS 가 19ms 늦게 깨웠다
    sent.append(now)
    last = now
    now += 0.0005
gaps = [b - a for a, b in zip(sent, sent[1:])]
check("늦은 프레임 다음도 최소 간격을 지킨다", min(gaps) >= MIN_GAP_S - 1e-9,
      f"최소 {min(gaps)*1000:.1f}ms (≥{MIN_GAP_S*1000:.0f}ms)")
# 절대 일정이라 프레임마다의 0.5ms 는 안 쌓인다. 19ms 늦은 것도 몇
# 프레임 뒤에는 제 일정으로 돌아와 있어야 한다.
check("늦은 만큼은 몇 프레임에 걸쳐 따라잡는다",
      abs(sent[-1] - 11 * I) < 0.002,
      f"마지막 {sent[-1]*1000:.1f}ms (제 일정 {11*I*1000:.0f}ms)")

print("\n  ✔ 재생 일정 이상 없음" if not fail else f"\n  ✗ {fail}건 실패")
sys.exit(1 if fail else 0)
