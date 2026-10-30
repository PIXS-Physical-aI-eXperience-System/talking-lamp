"""Decide when the user has left the desk and when to greet their return.

Vision already debounces face loss into ``/lamp/vision/presence``. That gate is
short so face following can react quickly. Leaving the desk needs a longer
wait, so a user who bends down to pick something up is not treated as gone.
Returning also needs a short wait: the away pose tilts the head up, and a
face seen for a moment while it moves must not end the away state.

After the greeting the user is often no longer in the middle of the frame, and
with the head still the face drifts out of view and S4 starts over. So face
following stays on for a short while once the greeting has played.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class Action:
    kind: str           # "center" when the user left, "greet" when they came back,
                        # "follow_on"/"follow_off" around the post-greeting gaze,
                        # "resume" to leave the away pose without a greeting


class S4Coordinator:
    """Return away/return effects. Time is passed in so tests need no clock."""

    def __init__(self, away_after_s: float = 20.0, return_after_s: float = 2.0,
                 greet_follow_s: float = 10.0) -> None:
        if away_after_s <= 0 or return_after_s < 0 or greet_follow_s < 0:
            raise ValueError("away_after_s must be positive, the others not negative")
        self.away_after_s = away_after_s
        self.return_after_s = return_after_s
        self.greet_follow_s = greet_follow_s
        self._follow_until: float | None = None
        self.away = False
        self._absent_since: float | None = None
        self._present_since: float | None = None
        self._present = False           # vision publishes presence only on change

    @property
    def greeting(self) -> bool:
        """The post-greeting gaze is following the face."""
        return self._follow_until is not None

    def presence(self, present: bool, now: float) -> list[Action]:
        self._present = present
        if not present:
            self._present_since = None
            if self._absent_since is None and not self.away:
                self._absent_since = now
            return []
        self._absent_since = None
        if not self.away:
            return []
        if self._present_since is None:
            self._present_since = now
        return self._confirm_return(now)

    def greeted(self, now: float) -> list[Action]:
        """The greeting finished playing. Follow the face for a while."""
        if self.away or self.greet_follow_s == 0:
            return []
        self._follow_until = now + self.greet_follow_s
        return [Action("follow_on")]

    def conversation_started(self, now: float) -> list[Action]:
        """A voice turn takes the head.

        End the post-greeting gaze early. A wake word while away means the
        user is back and talking already: leave the away pose, but skip the
        greeting, which would play over the reply.
        """
        if self.away:
            self.away = False
            self._present_since = None
            # No new presence message will come if the face is still not in
            # view, so start the absence wait here.
            self._absent_since = None if self._present else now
            return [Action("resume")]
        if self._follow_until is None:
            return []
        self._follow_until = None
        return [Action("follow_off")]

    def tick(self, now: float, *, busy: bool = False) -> list[Action]:
        """``busy``: another scenario holds the head, so the user is not gone."""
        if self.away:
            return self._confirm_return(now)
        if not self._present and self._absent_since is None:
            # Vision publishes presence only on change, so a node started
            # with nobody in view never hears "absent". Count from here.
            self._absent_since = now
        effects = []
        if self._follow_until is not None and now >= self._follow_until:
            self._follow_until = None
            effects.append(Action("follow_off"))
        if self._absent_since is None:
            return effects
        if busy:
            # A voice turn, a motion or a sound turn holds the head. Start the
            # wait again after it ends.
            self._absent_since = now
            return effects
        if now - self._absent_since < self.away_after_s:
            return effects
        self._absent_since = None
        self.away = True
        self._follow_until = None       # "center" turns following off itself
        return [Action("center")]

    def _confirm_return(self, now: float) -> list[Action]:
        if self._present_since is None or now - self._present_since < self.return_after_s:
            return []
        self._present_since = None
        self.away = False
        return [Action("greet")]
