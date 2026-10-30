"""Supervised three-step S2 camera-follow bench for the assembled lamp.

Run from the workstation while the Pi motion daemon is stopped and a person
supports the arm. Only base yaw and wrist pitch receive torque. Each pulse is
at most two degrees, and torque is released in a finally block. SSH passwords
come from SSHPASS; this file never stores them.
"""

from __future__ import annotations

import json
import os
from pathlib import Path
import shlex
import subprocess
import sys
import time
from types import SimpleNamespace

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
from vision.image_follow import ImageFaceFollower  # noqa: E402

JETSON = "asdf@100.79.117.124"
PI = "pixs@192.168.100.2"

CAPTURE = r'''
import cv2, json, sys
sys.path.insert(0, '/home/asdf/talking-lamp-integration/src')
from vision.detector import FaceDetector
cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*'MJPG'))
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1920)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 1080)
for _ in range(4): ok, frame = cap.read()
cap.release()
if not ok: raise RuntimeError('camera frame unavailable')
faces = FaceDetector('/home/asdf/vision-bench/models/yunet_2023mar.onnx')(frame)
print(json.dumps({'width':frame.shape[1], 'height':frame.shape[0],
                  'faces':[{'conf':f.conf, 'box':f.box} for f in faces]}))
'''

JOG = r'''
import json, subprocess, sys, time
sys.path.insert(0, '/home/pixs/talking-lamp/src')
from lerobot.motors import Motor, MotorNormMode
from lerobot.motors.feetech import FeetechMotorsBus
from motion.hardware_alignment import HardwareAlignment
if subprocess.run(['systemctl','is-active','--quiet','talking-lamp-motion.service']).returncode == 0:
    raise RuntimeError('motion daemon owns the servo bus')
deltas = {'base_yaw':float(sys.argv[1]), 'wrist_pitch':float(sys.argv[2])}
if any(abs(x) > 2.001 for x in deltas.values()): raise ValueError('step exceeds two degrees')
names = ('base_yaw','base_pitch','elbow_pitch','wrist_roll','wrist_pitch')
bus = FeetechMotorsBus(port='/dev/ttyACM0', motors={
    n:Motor(i,'sts3215',MotorNormMode.RANGE_M100_100) for i,n in enumerate(names,1)})
bus.connect()
active = tuple(deltas)
try:
    if any(bus.sync_read('Torque_Enable',normalize=False).values()):
        raise RuntimeError('another process has motor torque enabled')
    before = bus.sync_read('Present_Position',normalize=False)
    alignment = HardwareAlignment.load()
    targets = {}
    for name, delta in deltas.items():
        i = names.index(name)
        target = round(before[name] + alignment.direction_sign[i] * delta
                       * alignment.counts_per_revolution / 360)
        lo, hi = alignment.raw_ranges[i]
        if not lo <= target <= hi: raise ValueError(name + ' would exceed calibrated range')
        targets[name] = target
    for name in active:
        bus.write('Goal_Position',name,int(before[name]),normalize=False)
    for name in active:
        bus.write('Torque_Enable',name,1,normalize=False)
    time.sleep(.3)
    for name in active:
        bus.write('Goal_Position',name,int(targets[name]),normalize=False)
    time.sleep(1.2)
    after = bus.sync_read('Present_Position',motors=list(active),normalize=False)
    print(json.dumps({'before':{n:before[n] for n in active},
                      'target':targets,'after':after}))
finally:
    try:
        for name in active: bus.write('Torque_Enable',name,0,normalize=False)
    finally:
        bus.disconnect(disable_torque=False)
'''


def ssh(host: str, command: str, code: str) -> dict:
    args = ["sshpass", "-e", "ssh"]
    if host == PI:
        # The outer sshpass removes SSHPASS from its child environment, so the
        # jump-host process needs its own password argument.
        proxy = ("ProxyCommand=sshpass -p "
                 + shlex.quote(os.environ["SSHPASS"])
                 + " ssh -W %h:%p " + JETSON)
        args += ["-o", proxy]
    args += [host, command]
    result = subprocess.run(args, input=code, text=True, capture_output=True,
                            timeout=20, env=os.environ.copy())
    if result.returncode:
        raise RuntimeError(f"SSH to {host} failed: {result.stderr[-500:]}")
    return json.loads(result.stdout.splitlines()[-1])


def main() -> None:
    if not os.environ.get("SSHPASS"):
        raise RuntimeError("Set SSHPASS for the two authorized SSH accounts")
    follower = ImageFaceFollower()
    for index in range(3):
        frame = ssh(JETSON, "/home/asdf/vision-venv/bin/python -", CAPTURE)
        faces = [SimpleNamespace(**face) for face in frame["faces"]]
        step = follower.propose(faces, frame["width"], frame["height"],
                                time.monotonic())
        if step is None:
            print(json.dumps({"step": index, "result": "no safe correction",
                              "faces": len(faces)}))
            break
        print(json.dumps({"step": index, "face_center": step.face_center,
                          "yaw_deg": step.yaw_deg, "wrist_deg": step.wrist_deg}),
              flush=True)
        result = ssh(PI, "/home/pixs/talking-lamp/lelamp_runtime/.venv/bin/python - "
                     f"{step.yaw_deg:.4f} {step.wrist_deg:.4f}", JOG)
        print(json.dumps({"step": index, "motor": result}), flush=True)
    frame = ssh(JETSON, "/home/asdf/vision-venv/bin/python -", CAPTURE)
    print(json.dumps({"final_faces": frame["faces"]}), flush=True)


if __name__ == "__main__":
    main()
