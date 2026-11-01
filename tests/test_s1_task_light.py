import pytest

from lamp_interaction.light import light_command
from lamp_interaction.s1 import (TaskLightFlow, TaskLightRequest, ends_task_light,
                                 task_light_command)


@pytest.mark.parametrize(("text", "kind"), [
    ("픽스야 책 비춰줘", "place"),
    ("여기 좀 비춰 줄래?", "place"),
    ("책상 비춰", "place"),
    ("책 빗쳐줘", "place"),
    ("채 빗쳐줘", "place"),
    ("그만 비춰", "release"),
    ("이제 비추지 마", "release"),
    ("조명 치워", "release"),
    ("조명 원래대로 해", "release"),
])
def test_task_light_requests(text, kind):
    assert task_light_command(text) == TaskLightRequest(kind)


@pytest.mark.parametrize("text", ["불 켜줘", "그만", "고맙습니다.", "치워 줘"])
def test_not_task_light_requests(text):
    assert task_light_command(text) is None


def test_shine_is_not_a_light_switch():
    # "책 비춰줘" moves the arm; it must not also be read as "불 켜줘".
    assert light_command("책 비춰줘") is None


def test_flow_waits_for_the_conversation_then_aims_and_lights():
    flow = TaskLightFlow(give_up_s=40)
    flow.request()
    assert flow.tick(0, busy=True, aim="off") == []
    assert not flow.holds_head
    assert flow.tick(1, busy=False, aim="off") == ["start"]
    assert flow.aiming and flow.holds_head
    assert flow.tick(5, busy=False, aim="searching") == []
    assert flow.tick(9, busy=False, aim="centering") == []
    assert flow.tick(12, busy=False, aim="centered") == []
    assert flow.tick(14, busy=False, aim="reaching") == []
    assert flow.tick(18, busy=False, aim="placed") == ["lit"]
    assert flow.state == "lit"
    assert flow.tick(60, busy=False, aim="placed") == []
    assert flow.release()
    assert flow.state == "idle"


def test_a_failed_reach_still_lights_the_centred_book():
    flow = TaskLightFlow(give_up_s=40)
    flow.request()
    flow.tick(0, busy=False, aim="off")
    assert flow.tick(39, busy=False, aim="centered") == []
    assert flow.tick(45, busy=False, aim="reaching") == []     # found it: no give-up
    assert flow.tick(47, busy=False, aim="reach_failed") == ["lit"]


def test_flow_gives_up_when_no_book_turns_up():
    flow = TaskLightFlow(give_up_s=40)
    flow.request()
    flow.tick(0, busy=False, aim="off")
    actions = [a for t in range(1, 42) for a in flow.tick(t, busy=False, aim="searching")]
    assert actions == ["give_up"]
    assert flow.state == "idle"
    assert not flow.release()


def test_flow_keeps_centring_past_the_give_up_time_but_not_forever():
    flow = TaskLightFlow(give_up_s=40)
    flow.request()
    flow.tick(0, busy=False, aim="off")
    assert flow.tick(45, busy=False, aim="centering") == []
    assert flow.tick(91, busy=False, aim="centering") == ["give_up"]


@pytest.mark.parametrize(("text", "ends"), [
    ("픽스야", False),
    ("빅스야?", False),
    ("불 켜줘", False),
    ("불 좀 더 밝게", False),
    ("조명 어둡게 해줘", False),
    ("불 꺼줘", True),
    ("지금 몇 시야?", True),
    ("픽스야, 음악 틀어줘", True),
])
def test_asking_pixs_for_something_else_ends_the_task_light(text, ends):
    command = light_command(text)
    assert ends_task_light(text, command and command.kind) is ends
