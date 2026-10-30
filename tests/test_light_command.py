import pytest

from lamp_interaction.light import (
    DEFAULT_BRIGHTNESS, DEVICE_MAX_BRIGHTNESS, MIN_BRIGHTNESS, LightCommand, LightState,
    device_brightness, light_command)


@pytest.mark.parametrize(("text", "kind"), [
    ("픽스야 불 켜줘", "on"),
    ("불 좀 켜 줄래?", "on"),
    ("조명 켜", "on"),
    ("불 꺼줘", "off"),
    ("이제 불 끄자", "off"),
    ("불 좀 밝게 해줘", "brighter"),
    ("조명 밝게 켜 줘", "brighter"),
    ("불 좀 어둡게", "dimmer"),
    ("조명 낮춰 줘", "dimmer"),
])
def test_light_requests(text, kind):
    assert light_command(text) == LightCommand(kind)


@pytest.mark.parametrize("text", [
    "고맙습니다.",
    "켜줘",                      # no light word: could be anything
    "의자가 불편해서 꺼내 줘",
    "불 켜지 마",
    "불 켜고 꺼",                # unclear
    "화이팅. 화이팅. 화이팅.",
])
def test_not_light_requests(text):
    assert light_command(text) is None


def test_brightness_steps():
    state = LightState()
    assert state.apply(LightCommand("dimmer")) == 0.0       # off stays off
    assert state.apply(LightCommand("on")) == DEFAULT_BRIGHTNESS
    assert state.apply(LightCommand("brighter")) == 1.0
    assert state.apply(LightCommand("dimmer")) == 0.5
    assert state.apply(LightCommand("brighter")) == 1.0
    for _ in range(10):
        state.apply(LightCommand("dimmer"))
    assert state.brightness == MIN_BRIGHTNESS
    assert state.apply(LightCommand("off")) == 0.0
    assert state.apply(LightCommand("brighter")) == DEFAULT_BRIGHTNESS


def test_every_level_is_visible_under_the_device_ceiling():
    # The Pi clamps anything above its ceiling, so each step must stay under it
    # and still differ from the one before.
    state = LightState()
    levels = [state.apply(LightCommand("on"))]
    while levels[-1] > MIN_BRIGHTNESS:
        levels.append(state.apply(LightCommand("dimmer")))
    shown = [device_brightness(level) for level in levels]
    assert shown[0] == DEVICE_MAX_BRIGHTNESS
    assert shown == sorted(set(shown), reverse=True)
    assert shown == [0.08, 0.04, 0.02]
    assert state.apply(LightCommand("brighter")) == 0.5
