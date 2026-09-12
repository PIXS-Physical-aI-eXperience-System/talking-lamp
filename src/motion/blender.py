"""Motion blender - composites the L0..L3 layers into one joint-space target.

Each layer emits a `LayerOutput`:

* ``value``    - (5,) joint angles (``absolute``) or an offset (``additive``);
* ``weight``   - (5,) in [0, 1], how hard the layer claims each joint;
* ``gain``     - scalar in [0, 1], the layer's own crossfade envelope.

Compositing runs **low priority first**; each layer is laid over the running
result with ``alpha = gain * weight`` per joint, so a higher-priority layer at
full weight fully overrides the ones below it and a fading one hands authority
back smoothly. Additive layers (idle bob, expressive primitives) just add
``alpha * value`` so they ride on top of whatever base pose won.

The blender is pure/stateless given the layers; all timing state lives in the
layers (see `layers.py`).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Protocol

import numpy as np

from .config import NJ, REST_POSE


@dataclass
class LayerOutput:
    value: np.ndarray
    weight: np.ndarray
    gain: float = 1.0
    additive: bool = False

    @staticmethod
    def inactive() -> "LayerOutput":
        return LayerOutput(np.zeros(NJ), np.zeros(NJ), 0.0, additive=True)


@dataclass
class BlendContext:
    """State the blender shares with every layer for the current tick."""

    q_current: np.ndarray      # joints actually commanded last tick
    t: float
    dt: float


class MotionLayer(Protocol):
    name: str
    priority: int

    def update(self, ctx: BlendContext) -> LayerOutput: ...


@dataclass
class BlendTrace:
    q: np.ndarray
    per_layer: dict[str, np.ndarray] = field(default_factory=dict)   # effective alpha
    authority: dict[str, np.ndarray] = field(default_factory=dict)   # contribution


class MotionBlender:
    def __init__(self, layers: list[MotionLayer], rest_pose: np.ndarray | None = None) -> None:
        self.layers = sorted(layers, key=lambda ly: ly.priority)
        self.rest_pose = np.asarray(rest_pose if rest_pose is not None else REST_POSE, float)

    def add(self, layer: MotionLayer) -> None:
        self.layers.append(layer)
        self.layers.sort(key=lambda ly: ly.priority)

    def get(self, name: str) -> MotionLayer:
        for ly in self.layers:
            if ly.name == name:
                return ly
        raise KeyError(name)

    def compute(self, ctx: BlendContext) -> BlendTrace:
        q = self.rest_pose.copy()
        trace = BlendTrace(q=q)
        for ly in self.layers:
            out = ly.update(ctx)
            alpha = np.clip(out.gain, 0.0, 1.0) * np.clip(out.weight, 0.0, 1.0)
            if not np.any(alpha):
                trace.per_layer[ly.name] = np.zeros(NJ)
                continue
            before = q.copy()
            if out.additive:
                q = q + alpha * out.value
            else:
                q = (1.0 - alpha) * q + alpha * out.value
            trace.per_layer[ly.name] = alpha
            trace.authority[ly.name] = q - before
        trace.q = q
        return trace
