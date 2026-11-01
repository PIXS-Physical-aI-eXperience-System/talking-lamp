"""The Pi says once per boot that the lamp needs a few minutes to wake up."""

from pathlib import Path
import re
import wave


ROOT = Path(__file__).resolve().parents[1]
UNIT = ROOT / "deploy/pi/talking-lamp-boot-announce.service"
DEVICE_UNIT = ROOT / "deploy/pi/talking-lamp-device.service"
WAV = ROOT / "deploy/pi/boot-announce.wav"


def test_boot_announcement_runs_once_and_never_fails_the_boot():
    unit = UNIT.read_text()
    assert "Type=oneshot" in unit
    assert "ExecStart=-/usr/bin/aplay " in unit
    assert "[Install]\nWantedBy=multi-user.target" in unit


def test_boot_announcement_uses_the_device_speaker_and_shipped_wav():
    unit = UNIT.read_text()
    card = re.search(r"--alsa-card (\S+)", DEVICE_UNIT.read_text()).group(1)
    assert f"-D plughw:CARD={card},DEV=0" in unit
    assert "/home/pixs/talking-lamp/deploy/pi/boot-announce.wav" in unit
    with wave.open(str(WAV), "rb") as w:
        assert (w.getframerate(), w.getnchannels(), w.getsampwidth()) == (16000, 1, 2)
        assert 3 < w.getnframes() / 16000 < 15
