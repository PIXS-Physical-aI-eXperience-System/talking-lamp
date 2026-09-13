# LeLamp Runtime

![](./assets/images/Banner.png)

This repository holds the code for controlling LeLamp. The runtime provides a comprehensive control system for the robotic lamp, including motor control, recording/replay functionality, voice interaction, and testing capabilities.

[LeLamp](https://github.com/humancomputerlab/LeLamp) is an open source robot lamp based on [Apple's Elegnt](https://machinelearning.apple.com/research/elegnt-expressive-functional-movement), made by [[Human Computer Lab]](https://www.humancomputerlab.com/)

## Overview

LeLamp Runtime is a Python-based control system that interfaces with the hardware components of LeLamp including:

- Servo motors for articulated movement
- Audio system (microphone and speaker)
- RGB LED lighting
- Camera system
- Voice interaction capabilities

## Project Structure

```
lelamp_runtime/
├── main.py                 # Main runtime entry point
├── pyproject.toml         # Project configuration and dependencies
├── lelamp/                # Core package
│   ├── setup_motors.py    # Motor configuration and setup
│   ├── calibrate.py       # Motor calibration utilities
│   ├── list_recordings.py # List all recorded motor movements
│   ├── record.py          # Movement recording functionality
│   ├── replay.py          # Movement replay functionality
│   ├── follower/          # Follower mode functionality
│   ├── leader/            # Leader mode functionality
│   └── test/              # Hardware testing modules
└── uv.lock               # Dependency lock file
```

## Installation

### Prerequisites

- UV package manager
- Hardware components properly assembled (see main LeLamp documentation)

### Setup

1. Clone the runtime repository:

```bash
git clone https://github.com/humancomputerlab/lelamp_runtime.git
cd lelamp_runtime
```

2. Install UV (if not already installed):

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
```

3. Install dependencies:

```bash
# If on your personal computer
uv sync

# If on Raspberry Pi
uv sync --extra hardware
```

**Note**: For motor setup and control, LeLamp Runtime can run on your computer and you only need to run `uv sync`. For other functionality that connects to the head Pi (LED control, audio, camera), you need to install LeLamp Runtime on that Pi and run `uv sync --extra hardware`.

If you have LFS problems, run the following command:

```bash
GIT_LFS_SKIP_SMUDGE=1 uv sync
```

If your installation process is slow, use the following environment variable:

```bash
export UV_CONCURRENT_DOWNLOADS=1
```

### Dependencies

The runtime includes several key dependencies:

- **feetech-servo-sdk**: For servo motor control
- **lerobot**: Robotics framework integration
- **livekit-agents**: Real-time voice interaction
- **numpy**: Mathematical operations
- **sounddevice**: Audio input/output
- **adafruit-circuitpython-neopixel**: RGB LED control (hardware)
- **rpi-ws281x**: Raspberry Pi LED control (hardware)

## Core Functionality

Prior to following the instructions here, you should have an overview of how to control LeLamp through [this tutorial](https://github.com/humancomputerlab/LeLamp/blob/master/docs/5.%20LeLamp%20Control.md).

### 1. Motor Setup and Calibration

1. **Find the servo driver port**:

This command finds the port your motor driver is connected to.

```bash
uv run lerobot-find-port
```

2. **Setup motors with unique IDs**:

This command set up each motor of LeLamp with an unique ID.

```bash
uv run -m lelamp.setup_motors --id your_lamp_name --port the_port_found_in_previous_step
```

3. **Calibrate motors**:

This command calibrate your motors.

Follower motors만 장착한 LeLamp은 다음 명령을 사용합니다. `uv`는 가상환경 안의
`.venv/bin/uv`가 아니라 사용자 명령 경로에 설치되는 실행 파일입니다.

```bash
uv run -m lelamp.calibrate \
  --id lelamp --port the_port_found_in_previous_step \
  --follower-only
```

현재 `lamp-pi`처럼 `uv` 명령은 없고 런타임 가상환경이 이미 설치돼 있으면,
`lelamp_runtime` 디렉터리에서 `uv run` 대신 `.venv/bin/python -m`을 사용합니다.

The calibration process will:

- Calibrate follower mode without asking for an absent leader
- Ensure proper servo positioning and response
- Set baseline positions for accurate movement

Use the zero pose shown in the LeLamp CAD when prompted for the middle position.
During range recording, move `base_yaw` and `wrist_roll` only about 90 degrees
clockwise and counterclockwise from center. Move `base_pitch` (the motor directly
above `base_yaw`), `elbow_pitch`, and `wrist_pitch` through their full mechanically
safe ranges. If `base_pitch` was not swept through its full range, recalibrate.
Normal motion is bound to the checked-in `lelamp` calibration and rejects a
different ID, homing offset, motor ID, drive mode, or raw range before opening
the motor bus. After recalibration, update both
`lelamp/motor_tuning.py::EXPECTED_FOLLOWER_CALIBRATION` and
`sim/hardware_alignment.json` from the new saved calibration before replaying.

### 2. Unit Testing

The runtime includes comprehensive testing modules to verify all hardware components:

#### RGB LEDs

```bash
# Run with sudo for hardware access
sudo uv run -m lelamp.test.test_rgb
```

#### Audio System (Microphone and Speaker)

```bash
uv run -m lelamp.test.test_audio
```

#### Motors

```bash
uv run -m lelamp.test.test_motors \
  --id lelamp --port the_port_found_in_previous_step \
  --recording movement_sequence_name
```

Repeat `--recording` to test a chosen sequence, or use `--all` to play every
named motion except the long `idle` loop. Add `--hold` to keep the service and
motor torque active after the last motion:

```bash
uv run -m lelamp.test.test_motors \
  --id lelamp --port the_port_found_in_previous_step \
  --all --speed 0.6 --pause-seconds 2 --hold
```

Without `--hold`, the test disconnects after playback and releases motor torque.
The loaded pitch joints can then drop under the lamp head's weight; this is not
a calibration change. Press Ctrl-C when using `--hold` to disconnect and release
the motors.

At connect time, `base_pitch` and `elbow_pitch` use the STS3215 factory position
gain of 32 so they can overcome the assembled lamp's static load. The lightly
loaded yaw and wrist joints keep the original gain of 16 to avoid shaking.

The bundled recordings contain absolute joint positions captured on the source
lamp. Playback recenters each recording on the hardware-aligned home pose. It
keeps the official trajectory shape while reducing it to 35% yaw, 12% base
pitch, 15% elbow pitch, 35% wrist roll, and 25% wrist pitch. `base_pitch`
is reversed to match this lamp's physical assembly. A motor test
also returns to that home pose after every selected recording. This prevents a
recording such as `happy_wiggle` from leaving the assembled head aimed at the
desk because of the source lamp's different starting posture.

### 3. Record and Replay Episodes

One of LeLamp's key features is the ability to record and replay movement sequences:

#### Recording Movement

To record a movement sequence:

```bash
uv run -m lelamp.record --id your_lamp_name --port the_port_found_in_previous_step --name movement_sequence_name
```

This will:

- Put the lamp in recording mode
- Allow you to manually manipulate the lamp
- Save the movement data to a CSV file

#### Replaying Movement

To replay a recorded movement:

```bash
uv run -m lelamp.replay --id lelamp --port the_port_found_in_previous_step --name movement_sequence_name
```


Before a normal motor-runtime shutdown, the lamp moves to the captured `SLEEP_POSE`
and verifies the measured joint positions before releasing servo torque.

The replay system will:

- Load the movement data from the CSV file
- Execute the recorded movements with proper timing
- Reproduce the original motion sequence

Playback validates all five joint columns and normalized values before connecting
to the motors. It then approaches the first frame from the measured pose and
interpolates the recorded frames at a steady command rate:

```bash
uv run -m lelamp.replay \
  --id lelamp --port the_port_found_in_previous_step \
  --name movement_sequence_name \
  --fps 30 \
  --speed 1.0 \
  --transition-seconds 3.0 \
  --max-planned-step 2.0 \
```

You can tune these:

- `--speed`: `1.0` keeps the original recording duration; `0.5` takes twice as long.
- `--fps`: motor command rate. Keep this at `30` so slow motion remains smooth.
- `--transition-seconds`: time used to move from the current pose to the first frame.
- Playback smooths recorded frame noise while preserving the first frame, last
  frame, and total frame count.
- `--max-planned-step`: only subdivides a remaining unsafe jump above this
  amount; the default smoothed recordings stay below `2.0` and are not stretched.
- `--max-relative-target`: optional measured-position clamp. It is disabled by
  default because it adds a serial read to every frame and distorts recordings
  when a motor trails the trajectory. Validated, smoothed targets remain within
  the calibrated joint range.

To verify recordings before movement, run:

```bash
uv run -m lelamp.test.analyze_recordings \
  --recording movement_sequence_name --max-step 3.0
```

This analysis command never connects to the robot. It prints each joint's full
range and the largest jump between source frames.

#### Listing Recordings

To view all recordings for a specific lamp:

```bash
uv run -m lelamp.list_recordings --id your_lamp_name
```

This will display:

- All available recordings for the specified lamp
- File information including row count
- Recording names that can be used for replay

#### File Format

Recorded movements are saved as CSV files with the naming convention:
`{sequence_name}.csv`

## 4. Start upon boot

If you want to start LeLamp's voice app upon booting. Create a systemd service file:

```bash
sudo nano /etc/systemd/system/lelamp.service
```

Add this content:

```bash
ini[Unit]
Description=Lelamp Runtime Service
After=network.target

[Service]
Type=simple
User=pi
WorkingDirectory=/home/pi/lelamp_runtime
ExecStart=/usr/bin/sudo uv run main.py console
Restart=always
RestartSec=5

[Install]
WantedBy=multi-user.target
```

Then enable and start the service:

```bash
sudo systemctl daemon-reload
sudo systemctl enable lelamp.service
sudo systemctl start lelamp.service
```

For other service controls:

```bash
# Disable from starting on boot
sudo systemctl disable lelamp.service

# Stop the currently running service
sudo systemctl stop lelamp.service

# Check status (should show "disabled" and "inactive")
sudo systemctl status lelamp.service
```

Note: Boot time might vary with each run and extended usage (>1 hour) can burn the motors.

## Sample Apps

Sample apps to test LeLamp's capabilities.

### LiveKit Voice Agent

To run a conversational agent on LeLamp, create a .env file with the following content in the root of this directory in your Raspberry Pi.

```bash
OPENAI_API_KEY=
LIVEKIT_URL=
LIVEKIT_API_KEY=
LIVEKIT_API_SECRET=
```

On how to get LiveKit secrets, please refer to [LiveKit's guide](https://docs.livekit.io/agents/start/voice-ai/). Install LiveKit CLI, then you can run the following command:

```bash
lk app env -w
cat .env.local
```

This will automatically create an `.env.local` file for you, which contains all the secrets on LiveKit side.

On how to get OpenAI secrets, you can follow this [FAQ](https://help.openai.com/en/articles/4936850-where-do-i-find-my-openai-api-key).

Then you can run the agent app by:

```bash
# Only need to run this once
sudo uv run main.py download-files

# Pick one of the below
# For Discrete Animation Mode
sudo uv run main.py console

# For Smooth Animation Mode
sudo uv run smooth_animation.py console
```

The physical motion profile is intentionally bound to the `lelamp` ID. To use
another ID, recalibrate and update both calibration records described above;
changing only `main.py` is rejected before the motor bus enables torque.

## Contributing

This is an open-source project by Human Computer Lab. Contributions are welcome through the GitHub repository.

## Maintainers
Maintained by [Human Computer Lab](https://www.humancomputerlab.com).

## Acknowledgments & Sponsors
See [CONTRIBUTORS.md](./CONTRIBUTORS.md) for contributors and their roles.  
See [SPONSORS.md](./SPONSORS.md) for sponsor thanks and how to support the project.

## License

Check the main [LeLamp repository](https://github.com/humancomputerlab/LeLamp) for licensing information.
