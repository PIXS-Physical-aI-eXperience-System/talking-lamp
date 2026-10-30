"""D (vision): camera frames -> desk objects and faces in ``lamp_base``.

Pure-numpy parts (``camera``, ``geometry``, ``labels``, ``pipeline``) import
without OpenCV; ``detector`` and ``calibration`` need ``cv2`` and
``onnxruntime`` and import them lazily. See src/vision/README.md.
"""

from .camera import Intrinsics
from .geometry import FRAME_ID, Pose, desk_point, face_point
from .labels import COCO, DESK, KO, to_korean
from .pipeline import FaceTarget, ObjectTarget, VisionFrame, VisionPipeline, Workspace

__all__ = [
    "FRAME_ID", "Intrinsics", "Pose", "desk_point", "face_point",
    "COCO", "DESK", "KO", "to_korean",
    "FaceTarget", "ObjectTarget", "VisionFrame", "VisionPipeline", "Workspace",
]
