"""Talking Lamp motion stack (part E): L0-L3 layers, IK, online trajectory gen,
Kalman tracking, and the 100 Hz runtime that blends them into servo commands.

See docs/파트-분배.md 4.5 and docs/진행-순서.md (E, steps 6-10).
"""

from importlib import import_module

# Keep protocol/client imports usable on a stdlib-only diagnostic machine.
_EXPORT_MODULES = {
    **dict.fromkeys(("ACC_LIMIT", "CONTROL_DT", "CONTROL_HZ", "JERK_LIMIT",
                     "JOINT_NAMES", "NJ", "REST_POSE", "VEL_LIMIT"), "config"),
    **dict.fromkeys(("LayerOutput", "MotionBlender"), "blender"),
    **dict.fromkeys(("IdleConfig", "IdleMotion"), "idle"),
    **dict.fromkeys(("IKResult", "IKSolver"), "ik"),
    **dict.fromkeys(("TargetTrack", "TrackConfig"), "kalman"),
    **dict.fromkeys(("ArmKinematics", "HeadPose"), "kinematics"),
    **dict.fromkeys(("IdleLayer", "PrimitiveLayer", "TaskLightLayer", "TrackLayer"), "layers"),
    **dict.fromkeys(("CLIP_NAMES", "Primitive", "PrimitiveLibrary"), "primitives"),
    **dict.fromkeys(("MotionRuntime", "NullBackend", "StepState"), "runtime"),
    "TrajectoryGenerator": "trajectory",
}


def __getattr__(name):
    if name not in _EXPORT_MODULES:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    value = getattr(import_module(f".{_EXPORT_MODULES[name]}", __name__), name)
    globals()[name] = value
    return value


def __dir__():
    return sorted(set(globals()) | set(__all__))


__all__ = [
    "ACC_LIMIT", "CONTROL_DT", "CONTROL_HZ", "JERK_LIMIT", "JOINT_NAMES", "NJ",
    "REST_POSE", "VEL_LIMIT", "LayerOutput", "MotionBlender", "IdleConfig",
    "IdleMotion", "IKResult", "IKSolver", "TargetTrack", "TrackConfig",
    "ArmKinematics", "HeadPose", "IdleLayer", "PrimitiveLayer", "TaskLightLayer",
    "TrackLayer", "CLIP_NAMES", "Primitive", "PrimitiveLibrary", "MotionRuntime",
    "NullBackend", "StepState", "TrajectoryGenerator",
]
