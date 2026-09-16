import asyncio
from dataclasses import dataclass

from lamp_interaction.turn import TurnOrchestrator, TurnResponse, TurnResult


@dataclass
class Result:
    success: bool
    code: str = "completed"
    message: str = ""
    state: str = ""


class FakePorts:
    def __init__(self, *, audio=True, motion=True, center=True):
        self.audio_ok = audio
        self.motion_ok = motion
        self.center_ok = center
        self.events = []
        self.release = asyncio.Event()

    async def wait_orientation(self, speech_id, timeout):
        self.events.append(("orientation", speech_id))
        return Result(True, state="aligned")

    async def play_audio(self, response):
        self.events.append(("audio.start", response.speech_id))
        await self.release.wait()
        self.events.append(("audio.end", response.speech_id))
        return Result(self.audio_ok, "completed" if self.audio_ok else "audio_failed")

    async def play_motion(self, name):
        self.events.append(("motion.start", name))
        self.release.set()
        await asyncio.sleep(0)
        self.events.append(("motion.end", name))
        return Result(self.motion_ok, "completed" if self.motion_ok else "motion_failed")

    async def return_center(self):
        self.events.append(("center",))
        return Result(self.center_ok, "centered" if self.center_ok else "center_failed", state="centered")

    async def play_idle(self):
        self.events.append(("idle",))
        return Result(True)

    async def cancel_audio(self):
        self.events.append(("audio.cancel",))

    async def cancel_motion(self):
        self.events.append(("motion.cancel",))

    async def interrupt_motion(self):
        self.events.append(("motion.interrupt",))


def response():
    return TurnResponse(
        speech_id="20000000-0000-0000-0000-000000000002",
        motion_name="nod", audio_stream=object())


def test_turn_waits_for_alignment_runs_response_concurrently_then_centers_and_idles():
    async def scenario():
        ports = FakePorts()
        result = await TurnOrchestrator(ports).run(response())
        assert result == TurnResult(True, "completed", "")
        names = [event[0] for event in ports.events]
        assert names[0] == "orientation"
        assert set(names[1:3]) == {"audio.start", "motion.start"}
        assert names.index("center") > names.index("audio.end")
        assert names.index("center") > names.index("motion.end")
        assert names.index("idle") > names.index("center")

    asyncio.run(scenario())


def test_failed_audio_or_motion_never_automatically_centers_or_idles():
    async def scenario():
        for audio, motion in ((False, True), (True, False)):
            ports = FakePorts(audio=audio, motion=motion)
            result = await TurnOrchestrator(ports).run(response())
            assert not result.success
            assert not any(event[0] in {"center", "idle"} for event in ports.events)

    asyncio.run(scenario())


def test_center_failure_prevents_idle():
    async def scenario():
        ports = FakePorts(center=False)
        result = await TurnOrchestrator(ports).run(response())
        assert not result.success and result.code == "center_failed"
        assert not any(event[0] == "idle" for event in ports.events)

    asyncio.run(scenario())


def test_barge_in_cancels_both_responses_then_interrupts_without_centering():
    async def scenario():
        ports = FakePorts()
        await TurnOrchestrator(ports).barge_in()
        names = [event[0] for event in ports.events]
        assert set(names[:2]) == {"audio.cancel", "motion.cancel"}
        assert names[2] == "motion.interrupt"
        assert "center" not in names and "idle" not in names

    asyncio.run(scenario())
