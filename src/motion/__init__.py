"""Talking Lamp motion stack (part E): L0-L3 layers, IK, online trajectory gen,
Kalman tracking, and the 100 Hz runtime that blends them into servo commands.

See docs/파트-분배.md 4.5 and docs/진행-순서.md (E, steps 6-10).
"""

from .blender import LayerOutput, MotionBlender
from .config import (
    ACC_LIMIT,
    CONTROL_DT,
    CONTROL_HZ,
    JERK_LIMIT,
    JOINT_NAMES,
    NJ,
    REST_POSE,
    VEL_LIMIT,
)
from .idle import IdleConfig, IdleMotion
from .ik import IKResult, IKSolver
from .kalman import TargetTrack, TrackConfig
from .kinematics import ArmKinematics, HeadPose
from .layers import IdleLayer, PrimitiveLayer, TaskLightLayer, TrackLayer
from .primitives import CLIP_NAMES, Primitive, PrimitiveLibrary
from .runtime import MotionRuntime, NullBackend, StepState
from .trajectory import TrajectoryGenerator

__all__ = [
    "ACC_LIMIT", "CONTROL_DT", "CONTROL_HZ", "JERK_LIMIT", "JOINT_NAMES", "NJ",
    "REST_POSE", "VEL_LIMIT", "LayerOutput", "MotionBlender", "IdleConfig",
    "IdleMotion", "IKResult", "IKSolver", "TargetTrack", "TrackConfig",
    "ArmKinematics", "HeadPose", "IdleLayer", "PrimitiveLayer", "TaskLightLayer",
    "TrackLayer", "CLIP_NAMES", "Primitive", "PrimitiveLibrary", "MotionRuntime",
    "NullBackend", "StepState", "TrajectoryGenerator",
]
