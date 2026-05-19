import time
from enum import Enum


class DrowsinessState(Enum):
    ALERT = "ALERT"
    DROWSY = "DROWSY"
    NO_FACE = "NO_FACE"


class DrowsinessMonitor:

    def __init__(self):
        self._state = DrowsinessState.NO_FACE
        self._closed_frames = 0
        self._nod_frames = 0
        self._smoothed_ear = 0.0
        self._state_enter_time = time.time()

    def update(
        self,
        raw_ear,
        has_face,
        head_down,
        closed_threshold,
        open_threshold,
        drowsy_frames,
        nod_frames,
        smoothing_alpha,
    ):

        if not has_face:
            self._state = DrowsinessState.NO_FACE
            self._closed_frames = 0
            self._nod_frames = 0
            self._smoothed_ear = 0.0
            return

        # Exponential moving average for EAR
        if self._smoothed_ear == 0.0:
            self._smoothed_ear = raw_ear
        else:
            self._smoothed_ear = (
                smoothing_alpha * raw_ear
                + (1 - smoothing_alpha) * self._smoothed_ear
            )

        # Hysteresis: use different thresholds for closing vs opening
        ear_closed = self._smoothed_ear < closed_threshold
        ear_open = self._smoothed_ear > open_threshold

        # Drowsiness from eye closure OR head nodding
        is_drowsy_signal = ear_closed or head_down

        if is_drowsy_signal:
            if ear_closed:
                self._closed_frames += 1
            if head_down:
                self._nod_frames += 1

            if (
                self._closed_frames >= drowsy_frames
                or self._nod_frames >= nod_frames
            ):
                if self._state != DrowsinessState.DROWSY:
                    self._state_enter_time = time.time()

                self._state = DrowsinessState.DROWSY

        else:
            self._closed_frames = 0
            self._nod_frames = 0

            if self._state == DrowsinessState.DROWSY:
                self._state = DrowsinessState.ALERT
                self._state_enter_time = time.time()
            else:
                self._state = DrowsinessState.ALERT

    @property
    def state(self):
        return self._state

    @property
    def smoothed_ear(self):
        return self._smoothed_ear

    def reset(self):
        self._state = DrowsinessState.NO_FACE
        self._closed_frames = 0
        self._nod_frames = 0
        self._smoothed_ear = 0.0
        self._state_enter_time = time.time()
