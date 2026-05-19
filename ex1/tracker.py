# src/tracker.py

import threading
import time

import cv2
import mediapipe as mp

from mediapipe.tasks import python
from mediapipe.tasks.python import vision

from constants import (
    DETECTION_RESOLUTIONS,
    DETECTION_FPS_OPTIONS,
    DEBOUNCE_FRAMES,
    IDLE_TIMEOUT,
    LANDMARK_RADIUS,
    LEGS_CLOSED_FACTOR,
    LEGS_OPEN_FACTOR,
    MIN_DETECTION_CONFIDENCE,
    MIN_PRESENCE_CONFIDENCE,
    MIN_TRACKING_CONFIDENCE,
    MODEL_PATH,
    NUM_POSES,
    SKELETON_CONNECTIONS,
    STATE_TIMEOUT,
)

from state import DetectionState, ExerciseState


class JumpingJackTracker:

    def __init__(self):

        self.state = ExerciseState()

        self.show_landmarks = True
        self.show_skeleton = True

        self._detection_fps_index = 2
        self._detection_resolution_index = 1

        self.fps = 0
        self._frame_count = 0
        self._last_fps_time = time.time()

        self._lock = threading.Lock()
        self._stop_event = threading.Event()

        self._latest_frame = None
        self._latest_detection = "closed"
        self._latest_counting = False
        self._latest_counter = 0
        self._latest_fps = 0

        self.camera = cv2.VideoCapture(0)

        if not self.camera.isOpened():
            raise RuntimeError(
                "Could not open webcam."
            )

        self.camera.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        self.camera.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

        self.options = (
            vision.PoseLandmarkerOptions(
                base_options=(
                    python.BaseOptions(
                        model_asset_path=str(
                            MODEL_PATH
                        )
                    )
                ),
                running_mode=(
                    vision.RunningMode.VIDEO
                ),
                num_poses=NUM_POSES,
                min_pose_detection_confidence=(
                    MIN_DETECTION_CONFIDENCE
                ),
                min_pose_presence_confidence=(
                    MIN_PRESENCE_CONFIDENCE
                ),
                min_tracking_confidence=(
                    MIN_TRACKING_CONFIDENCE
                )
            )
        )

        self.landmarker = (
            vision.PoseLandmarker
            .create_from_options(
                self.options
            )
        )

        self._thread = threading.Thread(
            target=self._detection_loop,
            daemon=True
        )
        self._thread.start()

    @property
    def detection_fps(self):
        return DETECTION_FPS_OPTIONS[self._detection_fps_index]

    @property
    def detection_resolution_label(self):
        return list(DETECTION_RESOLUTIONS.keys())[self._detection_resolution_index]

    def increase_detection_fps(self):
        self._detection_fps_index = min(
            self._detection_fps_index + 1,
            len(DETECTION_FPS_OPTIONS) - 1
        )

    def decrease_detection_fps(self):
        self._detection_fps_index = max(
            self._detection_fps_index - 1,
            0
        )

    def increase_detection_resolution(self):
        self._detection_resolution_index = min(
            self._detection_resolution_index + 1,
            len(DETECTION_RESOLUTIONS) - 1
        )

    def decrease_detection_resolution(self):
        self._detection_resolution_index = max(
            self._detection_resolution_index - 1,
            0
        )

    def detect_jumping_jack_state(
        self,
        landmarks
    ):

        left_shoulder = landmarks[11]
        right_shoulder = landmarks[12]

        left_wrist = landmarks[15]
        right_wrist = landmarks[16]

        left_hip = landmarks[23]
        right_hip = landmarks[24]

        left_ankle = landmarks[27]
        right_ankle = landmarks[28]

        shoulder_y = (
            left_shoulder.y
            + right_shoulder.y
        ) / 2

        wrist_y = (
            left_wrist.y
            + right_wrist.y
        ) / 2

        arms_open = wrist_y < shoulder_y
        arms_closed = wrist_y > shoulder_y

        ankle_distance = abs(
            left_ankle.x
            - right_ankle.x
        )

        hip_distance = abs(
            left_hip.x
            - right_hip.x
        )

        legs_open = (
            ankle_distance
            > hip_distance
            * LEGS_OPEN_FACTOR
        )

        legs_closed = (
            ankle_distance
            < hip_distance
            * LEGS_CLOSED_FACTOR
        )

        if (
            arms_open
            and legs_open
        ):
            return DetectionState.OPEN

        if (
            arms_closed
            and legs_closed
        ):
            return DetectionState.CLOSED

        return DetectionState.UNDEFINED

    def draw_landmarks(
        self,
        frame,
        landmarks,
        width,
        height
    ):

        if self.show_landmarks:
            for landmark in landmarks:

                x = int(
                    landmark.x * width
                )

                y = int(
                    landmark.y * height
                )

                cv2.circle(
                    frame,
                    (x, y),
                    LANDMARK_RADIUS,
                    (0, 255, 0),
                    -1
                )

        if self.show_skeleton:
            for start_idx, end_idx in SKELETON_CONNECTIONS:
                start = landmarks[start_idx]
                end = landmarks[end_idx]

                x1 = int(start.x * width)
                y1 = int(start.y * height)
                x2 = int(end.x * width)
                y2 = int(end.y * height)

                cv2.line(
                    frame,
                    (x1, y1),
                    (x2, y2),
                    (0, 255, 0),
                    2
                )

    def _detection_loop(self):

        while not self._stop_event.is_set():

            loop_start = time.time()

            success, frame = self.camera.read()

            if not success:
                continue

            frame = cv2.flip(frame, 1)

            height, width, _ = frame.shape

            res_w, res_h = list(DETECTION_RESOLUTIONS.values())[
                self._detection_resolution_index
            ]

            small_frame = cv2.resize(frame, (res_w, res_h))

            rgb_frame = cv2.cvtColor(
                small_frame, cv2.COLOR_BGR2RGB
            )

            mp_image = mp.Image(
                image_format=mp.ImageFormat.SRGB,
                data=rgb_frame
            )

            timestamp_ms = int(time.time() * 1000)

            result = self.landmarker.detect_for_video(
                mp_image, timestamp_ms
            )

            if result.pose_landmarks:
                landmarks = result.pose_landmarks[0]

                self.draw_landmarks(
                    frame, landmarks, width, height
                )

                raw_state = self.detect_jumping_jack_state(landmarks)

                self.state.update(
                    raw_state,
                    DEBOUNCE_FRAMES,
                    STATE_TIMEOUT,
                    IDLE_TIMEOUT,
                )

            else:
                self.state.update(
                    DetectionState.NO_PERSON,
                    DEBOUNCE_FRAMES,
                    STATE_TIMEOUT,
                    IDLE_TIMEOUT,
                )

            self._frame_count += 1
            now = time.time()
            if now - self._last_fps_time >= 1.0:
                self.fps = self._frame_count / (now - self._last_fps_time)
                self._frame_count = 0
                self._last_fps_time = now

            with self._lock:
                self._latest_frame = frame.copy()
                self._latest_detection = self.state.detection
                self._latest_counting = self.state.is_counting
                self._latest_counter = self.state.counter
                self._latest_fps = self.fps

            elapsed = time.time() - loop_start
            target_interval = 1.0 / self.detection_fps
            if elapsed < target_interval:
                time.sleep(target_interval - elapsed)

    def update(self):

        with self._lock:
            if self._latest_frame is None:
                return False, None, DetectionState.NO_PERSON, False, 0, 0

            return (
                True,
                self._latest_frame,
                self._latest_detection,
                self._latest_counting,
                self._latest_counter,
                self._latest_fps,
            )

    def release(self):

        self._stop_event.set()
        self._thread.join(timeout=2.0)
        self.camera.release()
        self.landmarker.close()

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.release()
