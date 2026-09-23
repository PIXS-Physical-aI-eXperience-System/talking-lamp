import math

import pytest

from lamp_device_bridge.expressions import (
    ExpressionError,
    expression_names,
    expression_payload,
    get_expression,
    request_expression,
)


EXPECTED_EXPRESSIONS = (
    "neutral",
    "happy",
    "excited",
    "sad",
    "angry",
    "surprised",
    "curious",
    "thinking",
    "shy",
    "love",
)


def test_catalog_contains_ten_distinct_8x8_rgb_expressions():
    assert expression_names() == EXPECTED_EXPRESSIONS

    frames = [get_expression(name).rgb for name in EXPECTED_EXPRESSIONS]
    assert all(len(frame) == 8 * 8 * 3 for frame in frames)
    assert all(any(frame) for frame in frames)
    assert all(all(0 <= channel <= 255 for channel in frame) for frame in frames)
    assert len(set(frames)) == len(EXPECTED_EXPRESSIONS)


def test_expression_payload_reuses_existing_led_frame_protocol():
    expression = get_expression("happy")

    payload = expression_payload("happy", 0.08)

    assert payload == {"rgb": list(expression.rgb), "brightness": 0.08}


@pytest.mark.parametrize("name", ["", "happiness", "HAPPY", "../happy"])
def test_unknown_expression_is_rejected_before_transport(name):
    calls = []
    with pytest.raises(ExpressionError) as error:
        request_expression(lambda *args: calls.append(args), name, 0.05)
    assert error.value.code == "unknown_expression"
    assert calls == []


def test_expression_request_uses_the_existing_frame_command():
    calls = []

    result = request_expression(
        lambda kind, payload: calls.append((kind, payload)) or {"state": "completed"},
        "neutral", 0.05,
    )

    assert result == {"state": "completed"}
    assert calls == [("led.frame", expression_payload("neutral", 0.05))]


@pytest.mark.parametrize("brightness", [-0.01, 1.01, math.nan, True])
def test_expression_brightness_must_be_a_finite_fraction(brightness):
    with pytest.raises(ExpressionError) as error:
        expression_payload("neutral", brightness)
    assert error.value.code == "invalid_brightness"
