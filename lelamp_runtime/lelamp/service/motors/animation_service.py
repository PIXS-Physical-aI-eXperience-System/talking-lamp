import os
import threading
import time
from typing import Any, Dict, List, Optional

from lelamp.follower import LeLampFollower, LeLampFollowerConfig
from lelamp.playback import (
    limit_action_steps,
    load_recording,
    park_and_disconnect,
    recording_path,
    retarget_actions,
    resample_actions,
    smooth_actions,
    transition_actions,
)


DEFAULT_MAX_PLANNED_STEP: float = 2.0
DEFAULT_MAX_RELATIVE_TARGET: float | None = None


class AnimationService:
    def __init__(
        self,
        port: str,
        lamp_id: str,
        fps: float = 30.0,
        source_fps: float = 30.0,
        speed: float = 1.0,
        duration: float = 5.0,
        idle_recording: str = "idle",
        max_planned_step: float = DEFAULT_MAX_PLANNED_STEP,
        max_relative_target: int | float | None = DEFAULT_MAX_RELATIVE_TARGET,
    ):
        self.port = port
        self.lamp_id = lamp_id
        self.fps = fps
        self.source_fps = source_fps
        self.speed = speed
        self.duration = duration
        self.idle_recording = idle_recording
        self.max_step = max_planned_step
        self.robot_config = LeLampFollowerConfig(
            port=port,
            id=lamp_id,
            max_relative_target=max_relative_target,
        )
        self.robot: Optional[LeLampFollower] = None
        self.recordings_dir = os.path.join(
            os.path.dirname(__file__), "..", "..", "recordings"
        )

        self._recording_cache: Dict[str, List[Dict[str, float]]] = {}
        self._current_state: Optional[Dict[str, float]] = None
        self._current_recording: Optional[str] = None
        self._current_frame_index = 0
        self._current_actions: List[Dict[str, float]] = []

        self._running = threading.Event()
        self._event_queue: list[tuple[str, Any]] = []
        self._event_lock = threading.Lock()
        self._event_thread: Optional[threading.Thread] = None

    def start(self):
        self.robot = LeLampFollower(self.robot_config)
        try:
            self.robot.connect(calibrate=False)
            self._current_state = self.robot.get_observation()
        except Exception:
            self.robot = None
            raise
        print(f"Animation service connected to {self.port}")

        self._running.set()
        self._event_thread = threading.Thread(target=self._event_loop, daemon=True)
        self._event_thread.start()
        self.dispatch("play", self.idle_recording)

    def stop(self, timeout: float = 5.0):
        self._running.clear()
        if self._event_thread and self._event_thread.is_alive():
            self._event_thread.join(timeout=timeout)
        if self._event_thread and self._event_thread.is_alive():
            print("Animation worker is still active; leaving the port connected")
            return

        if self.robot:
            park_and_disconnect(self.robot)
            self.robot = None

    def dispatch(self, event_type: str, payload: Any):
        """Dispatch an event, using the same interface as ServiceBase."""
        if not self._running.is_set():
            print(f"Animation service is not running, ignoring event {event_type}")
            return

        with self._event_lock:
            self._event_queue.append((event_type, payload))

    def _event_loop(self):
        while self._running.is_set():
            with self._event_lock:
                if self._event_queue:
                    event_type, payload = self._event_queue.pop(0)
                else:
                    event_type, payload = None, None

            if event_type:
                try:
                    self.handle_event(event_type, payload)
                except Exception as exc:
                    print(f"Error handling event {event_type}: {exc}")

            self._continue_playback()
            time.sleep(1.0 / self.fps)

    def handle_event(self, event_type: str, payload: Any):
        if event_type == "play":
            self._handle_play(payload)
        else:
            print(f"Unknown event type: {event_type}")

    def _handle_play(self, recording_name: str):
        """Start a time-scaled recording after a smooth pose transition."""
        if not self.robot:
            print("Robot not connected")
            return

        actions = self._load_recording(recording_name)
        if not actions:
            return

        print(f"Starting {recording_name} at {self.speed:g}x speed")
        if self._current_state is not None:
            transition = transition_actions(
                self._current_state,
                actions[0],
                command_fps=self.fps,
                duration=self.duration,
            )
            actions = limit_action_steps(
                transition + actions[1:],
                start_pose=self._current_state,
                max_step=self.max_step,
            )

        self._current_recording = recording_name
        self._current_actions = actions
        self._current_frame_index = 0

    def _continue_playback(self):
        if not self._current_recording or not self._current_actions or not self.robot:
            return

        try:
            if self._current_frame_index < len(self._current_actions):
                action = self._current_actions[self._current_frame_index]
                sent_action = self.robot.send_action(action)
                self._current_state = sent_action.copy()
                self._current_frame_index += 1
                return

            self._handle_play(self.idle_recording)
        except Exception as exc:
            print(f"Error in playback: {exc}")
            self._current_recording = None
            self._current_actions = []
            self._current_frame_index = 0

    def get_available_recordings(self) -> List[str]:
        if not os.path.exists(self.recordings_dir):
            return []

        return sorted(
            filename[:-4]
            for filename in os.listdir(self.recordings_dir)
            if filename.endswith(".csv")
        )

    def _load_recording(
        self, recording_name: str
    ) -> Optional[List[Dict[str, float]]]:
        if recording_name in self._recording_cache:
            return self._recording_cache[recording_name]

        try:
            csv_path = recording_path(self.recordings_dir, recording_name)
            source_actions = smooth_actions(
                retarget_actions(load_recording(csv_path))
            )
            actions = resample_actions(
                source_actions,
                source_fps=self.source_fps,
                command_fps=self.fps,
                speed=self.speed,
            )
            self._recording_cache[recording_name] = actions
            return actions
        except Exception as exc:
            print(f"Error loading recording {recording_name}: {exc}")
            return None
