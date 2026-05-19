import time
from enum import Enum


class DetectionState(Enum):
    OPEN = "open"
    CLOSED = "closed"
    NO_PERSON = "no_person"
    UNDEFINED = "undefined"


class ExerciseState:

    def __init__(self):
        self.counter = 0
        self.is_counting = False
        self._detection = DetectionState.CLOSED
        self._pending_state = None
        self._pending_frames = 0
        self._state_enter_time = time.time()
        self._idle_time = None

    def update(self, detection: DetectionState, debounce_frames: int, state_timeout: float, idle_timeout: float):

        if detection == DetectionState.NO_PERSON:
            self._pending_state = None
            self._pending_frames = 0
            self._detection = DetectionState.NO_PERSON
            self.is_counting = False
            self._idle_time = None
            return

        if detection == DetectionState.UNDEFINED:
            return

        if self._detection == DetectionState.NO_PERSON:
            self._detection = detection
            self._idle_time = None
            self._pending_state = None
            self._pending_frames = 0

        elif detection == self._detection:
            self._pending_state = None
            self._pending_frames = 0

        elif detection == self._pending_state:
            self._pending_frames += 1

            if self._pending_frames >= debounce_frames:
                if (
                    self._detection == DetectionState.CLOSED
                    and self._pending_state == DetectionState.OPEN
                ):
                    self._detection = DetectionState.OPEN
                    self._idle_time = None
                    self._state_enter_time = time.time()

                elif (
                    self._detection == DetectionState.OPEN
                    and self._pending_state == DetectionState.CLOSED
                ):
                    if time.time() - self._state_enter_time < state_timeout:
                        self.counter += 1
                        self.is_counting = True
                        self._idle_time = None

                    self._detection = DetectionState.CLOSED

                self._pending_state = None
                self._pending_frames = 0

        else:
            self._pending_state = detection
            self._pending_frames = 1

        if self.is_counting and self._detection == DetectionState.CLOSED:
            if self._idle_time is None:
                self._idle_time = time.time()
            elif time.time() - self._idle_time >= idle_timeout:
                self.is_counting = False

    @property
    def detection(self):
        return self._detection

    def reset(self):
        self.counter = 0
        self.is_counting = False
        self._detection = DetectionState.CLOSED
        self._pending_state = None
        self._pending_frames = 0
        self._state_enter_time = time.time()
        self._idle_time = None
