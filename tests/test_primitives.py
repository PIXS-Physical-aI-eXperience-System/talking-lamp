import sys
from pathlib import Path

import numpy as np
import pytest

from motion.config import NJ
from motion.hardware_alignment import HardwareAlignment
from motion.primitives import CLIP_NAMES, Primitive, PrimitiveLibrary


RUNTIME_ROOT = Path(__file__).resolve().parents[1] / "lelamp_runtime"
sys.path.insert(0, str(RUNTIME_ROOT))

from lelamp.playback import (  # noqa: E402
    JOINT_KEYS,
    load_recording,
    retarget_actions,
    smooth_actions,
)


@pytest.mark.parametrize("name", CLIP_NAMES)
def test_every_clip_loads(name):
    p = Primitive.load(name)
    assert p.offsets.shape[1] == NJ
    assert p.times[0] == 0.0
    assert p.duration > 0.5
    # relative: starts at zero offset
    assert np.allclose(p.offsets[0], 0.0)


def test_sample_is_zero_outside_clip():
    p = Primitive.load("nod")
    assert np.allclose(p.sample(-1.0), 0.0)
    assert np.allclose(p.sample(p.duration + 0.5), 0.0)
    assert np.linalg.norm(p.sample(p.duration / 2)) > 0.0


def test_idle_clip_loops():
    p = Primitive.load("idle")
    assert p.loop
    a = p.sample(1.0)
    b = p.sample(1.0 + p.duration)
    assert np.allclose(a, b, atol=1e-6)


def test_sign_and_scale_applied():
    unit = Primitive.load("nod", sign=np.ones(NJ), scale=np.ones(NJ))
    sign = np.array([1, -1, 1, 1, 1.0])
    scale = np.full(NJ, 0.5)
    overridden = Primitive.load("nod", sign=sign, scale=scale)

    assert np.allclose(overridden.offsets, unit.offsets * sign * scale, atol=1e-12)


@pytest.mark.parametrize("name", CLIP_NAMES)
def test_default_offsets_match_smoothed_runtime_retargeting_in_simulation_radians(name):
    primitive = Primitive.load(name)
    recording_path = RUNTIME_ROOT / "lelamp" / "recordings" / f"{name}.csv"
    retargeted = smooth_actions(
        retarget_actions(load_recording(recording_path)), alpha=0.15
    )
    normalized = np.asarray(
        [[frame[joint] for joint in JOINT_KEYS] for frame in retargeted],
        dtype=float,
    )
    simulation_radians = HardwareAlignment.load().normalized_to_radians(normalized)
    expected_offsets = simulation_radians - simulation_radians[0]

    assert np.allclose(primitive.offsets, expected_offsets, atol=1e-12)


def test_default_offsets_reverse_base_pitch_like_runtime_retargeting():
    primitive = Primitive.load("nod")

    assert primitive.offsets[-1, 1] < 0.0


def test_resample_to_control_rate():
    p = Primitive.load("nod").resampled(0.01)
    assert np.allclose(np.diff(p.times), 0.01, atol=1e-3)
    assert np.allclose(p.sample(0.0), 0.0)


def test_library_caches_and_lists():
    lib = PrimitiveLibrary()
    assert lib.get("nod") is lib.get("nod")
    assert "headshake" in lib.available()


def test_library_honors_scale_after_default_load_without_changing_cached_default():
    lib = PrimitiveLibrary()
    default = lib.get("nod")
    muted = lib.get("nod", scale=np.zeros(NJ))
    np.testing.assert_array_equal(muted.offsets, np.zeros_like(muted.offsets))
    assert np.linalg.norm(default.offsets) > 0
    assert lib.get("nod") is default


def test_library_honors_direction_after_default_load():
    lib = PrimitiveLibrary()
    default = lib.get("nod")
    reversed_clip = lib.get("nod", sign=np.array([-1, 1, -1, -1, -1]))
    np.testing.assert_allclose(reversed_clip.offsets, -default.offsets, atol=1e-12)


def test_library_honors_loop_after_default_load():
    lib = PrimitiveLibrary()
    default = lib.get("nod")
    looped = lib.get("nod", loop=True)
    t = default.duration / 2
    np.testing.assert_allclose(looped.sample(looped.duration + t), default.sample(t))
    assert np.linalg.norm(looped.sample(looped.duration + t)) > 0
    np.testing.assert_array_equal(default.sample(default.duration + t), np.zeros(NJ))
