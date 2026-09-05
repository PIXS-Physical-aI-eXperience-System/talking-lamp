import numpy as np
import pytest

from motion.config import NJ
from motion.primitives import CLIP_NAMES, Primitive, PrimitiveLibrary


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
    base = Primitive.load("nod")
    flipped = Primitive.load("nod", sign=np.array([1, -1, 1, 1, 1.0]))
    half = Primitive.load("nod", scale=np.full(NJ, 0.5))
    t = base.duration / 3
    assert np.allclose(base.sample(t)[1], -flipped.sample(t)[1])
    assert np.allclose(base.sample(t) * 0.5, half.sample(t), atol=1e-9)


def test_resample_to_control_rate():
    p = Primitive.load("nod").resampled(0.01)
    assert np.allclose(np.diff(p.times), 0.01, atol=1e-3)
    assert np.allclose(p.sample(0.0), 0.0)


def test_library_caches_and_lists():
    lib = PrimitiveLibrary()
    assert lib.get("nod") is lib.get("nod")
    assert "headshake" in lib.available()
