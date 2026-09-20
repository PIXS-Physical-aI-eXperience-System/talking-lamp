import os
from typing import Any, List, Optional
from ..base import ServiceBase
from lelamp.follower import LeLampFollowerConfig, LeLampFollower
from lelamp.playback import (
    HOME_POSE,
    load_recording,
    park_and_disconnect,
    play_actions,
    recording_path,
    retarget_actions,
)

DEFAULT_MAX_PLANNED_STEP: float = 2.0
DEFAULT_MAX_RELATIVE_TARGET: float | None = None


class MotorsService(ServiceBase):
    def __init__(
        self,
        port: str,
        lamp_id: str,
        fps: float = 30.0,
        source_fps: float = 30.0,
        speed: float = 1.0,
        transition_seconds: float = 3.0,
        max_planned_step: float = DEFAULT_MAX_PLANNED_STEP,
        max_relative_target: int | float | None = DEFAULT_MAX_RELATIVE_TARGET,
    ):
        super().__init__("motors")
        self.port = port
        self.lamp_id = lamp_id
        self.fps = fps
        self.source_fps = source_fps
        self.speed = speed
        self.transition_seconds = transition_seconds
        self.max_step = max_planned_step
        self.robot_config = LeLampFollowerConfig(
            port=port,
            id=lamp_id,
            max_relative_target=max_relative_target,
        )
        self.robot: Optional[LeLampFollower] = None
        self.last_error: Optional[Exception] = None
        self.recordings_dir = os.path.join(
            os.path.dirname(__file__), "..", "..", "recordings"
        )
    
    def start(self):
        if self.robot is not None:
            raise RuntimeError("Robot is still owned; stop it safely before restarting")
        self.robot = LeLampFollower(self.robot_config)
        try:
            self.robot.connect(calibrate=False)
            super().start()
        except BaseException:
            # Preserve ownership on a cleanup failure; stop() only clears
            # the robot after confirmed parking and torque release.
            self.stop()
            raise
        self.logger.info(f"Motors service connected to {self.port}")

    def stop(self, timeout: float = 5.0):
        super().stop(timeout)
        if self._worker_thread and self._worker_thread.is_alive():
            self.logger.error("Motor worker is still active; leaving the port connected")
            return
        if self.robot:
            if self.robot.is_connected or self.robot.bus.is_connected:
                park_and_disconnect(self.robot)
            self.robot = None
    
    def handle_event(self, event_type: str, payload: Any):
        if event_type == "play":
            self._handle_play(payload)
        else:
            self.logger.warning(f"Unknown event type: {event_type}")
    
    def _handle_play(self, recording_name: str):
        """Play a recording by name"""
        if not self.robot:
            self.logger.error("Robot not connected")
            return
        
        self.last_error = None
        try:
            csv_path = recording_path(self.recordings_dir, recording_name)
            actions = retarget_actions(load_recording(csv_path))
            self.logger.info(
                f"Playing {len(actions)} source frames from {recording_name} "
                f"at {self.speed:g}x speed"
            )
            report = play_actions(
                self.robot,
                actions,
                source_fps=self.source_fps,
                command_fps=self.fps,
                speed=self.speed,
                transition_seconds=self.transition_seconds,
                return_pose=HOME_POSE,
                return_seconds=self.transition_seconds,
                max_step=self.max_step,
                should_stop=self._stop_event.is_set,
            )
            if report.interrupted:
                self.logger.info(f"Stopped playback: {recording_name}")
                return
            if report.clipped_frames:
                details = ", ".join(
                    f"{joint}={count}"
                    for joint, count in report.clipped_joints.items()
                )
                self.logger.warning(
                    f"Safety limit clipped {report.clipped_frames} frames "
                    f"while playing {recording_name}: {details}"
                )
            self.logger.info(f"Finished playing recording: {recording_name}")
            
        except Exception as exc:
            self.last_error = exc
            self.logger.error(f"Error playing recording {recording_name}: {exc}")
    
    def get_available_recordings(self) -> List[str]:
        """Get list of recording names available for this lamp ID"""
        if not os.path.exists(self.recordings_dir):
            return []
        
        recordings = []
        suffix = ".csv"
        
        for filename in os.listdir(self.recordings_dir):
            if filename.endswith(suffix):
                # Remove the CSV suffix to get the recording name.
                recording_name = filename[:-len(suffix)]
                recordings.append(recording_name)
        
        return sorted(recordings)
