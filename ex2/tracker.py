import math
import threading
import time

import cv2
import mediapipe as mp

from mediapipe.tasks import python
from mediapipe.tasks.python import vision

from constants import (
    DETECTION_RESOLUTIONS,
    DETECTION_FPS_OPTIONS,
    DROWSY_FRAME_THRESHOLD,
    EAR_CLOSED_THRESHOLD,
    EAR_OPEN_THRESHOLD,
    EAR_SMOOTHING_ALPHA,
    FACE_KEYPOINTS,
    HEAD_DOWN_THRESHOLD,
    HEAD_NOD_FRAMES,
    HEAD_TILT_THRESHOLD,
    LANDMARK_RADIUS,
    MIN_DETECTION_CONFIDENCE,
    MIN_PRESENCE_CONFIDENCE,
    MIN_TRACKING_CONFIDENCE,
    MODEL_PATH,
    NUM_FACES,
)

from state import DrowsinessState, DrowsinessMonitor


class DrowsinessTracker:

    def __init__(self):

        self.monitor = DrowsinessMonitor()

        self.show_landmarks = True
        self.show_eyes = True

        self._detection_fps_index = 2
        self._detection_resolution_index = 1

        self.fps = 0
        self._frame_count = 0
        self._last_fps_time = time.time()

        self._lock = threading.Lock()
        self._stop_event = threading.Event()

        self._latest_frame = None
        self._latest_state = DrowsinessState.NO_FACE
        self._latest_ear = 0.0
        self._latest_head_tilt = 0.0
        self._latest_fps = 0

        self.camera = cv2.VideoCapture(0)

        if not self.camera.isOpened():
            raise RuntimeError(
                "Could not open webcam."
            )

        self.camera.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        self.camera.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

        self.options = (
            vision.FaceLandmarkerOptions(
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
                num_faces=NUM_FACES,
                min_face_detection_confidence=(
                    MIN_DETECTION_CONFIDENCE
                ),
                min_face_presence_confidence=(
                    MIN_PRESENCE_CONFIDENCE
                ),
                min_tracking_confidence=(
                    MIN_TRACKING_CONFIDENCE
                ),
                output_face_blendshapes=True,
            )
        )

        self.landmarker = (
            vision.FaceLandmarker
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

    @staticmethod
    def _distance(p1, p2):
        return math.sqrt(
            (p1.x - p2.x) ** 2
            + (p1.y - p2.y) ** 2
        )

    def calculate_ear(self, landmarks):

        left_inner = landmarks[FACE_KEYPOINTS["left_eye_inner"]]
        left_outer = landmarks[FACE_KEYPOINTS["left_eye_outer"]]
        left_upper1 = landmarks[FACE_KEYPOINTS["left_eye_upper1"]]
        left_upper2 = landmarks[FACE_KEYPOINTS["left_eye_upper2"]]
        left_lower1 = landmarks[FACE_KEYPOINTS["left_eye_lower1"]]
        left_lower2 = landmarks[FACE_KEYPOINTS["left_eye_lower2"]]

        right_inner = landmarks[FACE_KEYPOINTS["right_eye_inner"]]
        right_outer = landmarks[FACE_KEYPOINTS["right_eye_outer"]]
        right_upper1 = landmarks[FACE_KEYPOINTS["right_eye_upper1"]]
        right_upper2 = landmarks[FACE_KEYPOINTS["right_eye_upper2"]]
        right_lower1 = landmarks[FACE_KEYPOINTS["right_eye_lower1"]]
        right_lower2 = landmarks[FACE_KEYPOINTS["right_eye_lower2"]]

        left_vertical = (
            self._distance(left_upper1, left_lower1)
            + self._distance(left_upper2, left_lower2)
        ) / 2.0

        left_horizontal = self._distance(left_inner, left_outer)

        left_ear = left_vertical / left_horizontal if left_horizontal > 0 else 0.0

        right_vertical = (
            self._distance(right_upper1, right_lower1)
            + self._distance(right_upper2, right_lower2)
        ) / 2.0

        right_horizontal = self._distance(right_inner, right_outer)

        right_ear = right_vertical / right_horizontal if right_horizontal > 0 else 0.0

        return (left_ear + right_ear) / 2.0

    def calculate_head_tilt(self, landmarks):

        nose = landmarks[FACE_KEYPOINTS["nose_tip"]]
        chin = landmarks[FACE_KEYPOINTS["chin"]]
        left_ear = landmarks[FACE_KEYPOINTS["left_ear"]]
        right_ear = landmarks[FACE_KEYPOINTS["right_ear"]]

        ear_mid_x = (left_ear.x + right_ear.x) / 2.0
        ear_mid_y = (left_ear.y + right_ear.y) / 2.0

        dx = nose.x - ear_mid_x
        dy = nose.y - ear_mid_y

        angle = math.atan2(dy, dx)

        tilt = abs(angle - math.pi / 2)

        nose_chin_dy = abs(nose.y - chin.y)

        head_down = nose_chin_dy < HEAD_DOWN_THRESHOLD

        return tilt, head_down

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

        if self.show_eyes:
            eye_indices = [
                FACE_KEYPOINTS["left_eye_inner"],
                FACE_KEYPOINTS["left_eye_outer"],
                FACE_KEYPOINTS["left_eye_upper1"],
                FACE_KEYPOINTS["left_eye_upper2"],
                FACE_KEYPOINTS["left_eye_lower1"],
                FACE_KEYPOINTS["left_eye_lower2"],
                FACE_KEYPOINTS["right_eye_inner"],
                FACE_KEYPOINTS["right_eye_outer"],
                FACE_KEYPOINTS["right_eye_upper1"],
                FACE_KEYPOINTS["right_eye_upper2"],
                FACE_KEYPOINTS["right_eye_lower1"],
                FACE_KEYPOINTS["right_eye_lower2"],
            ]

            for idx in eye_indices:
                lm = landmarks[idx]
                x = int(lm.x * width)
                y = int(lm.y * height)

                cv2.circle(
                    frame,
                    (x, y),
                    LANDMARK_RADIUS + 2,
                    (0, 0, 255),
                    -1
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

            has_face = False
            ear_value = 0.0
            head_tilt = 0.0

            if result.face_landmarks:
                landmarks = result.face_landmarks[0]

                has_face = True

                ear_value = self.calculate_ear(landmarks)
                head_tilt, head_down = self.calculate_head_tilt(landmarks)

                self.draw_landmarks(
                    frame, landmarks, width, height
                )

                self.monitor.update(
                    ear_value,
                    has_face,
                    head_down,
                    EAR_CLOSED_THRESHOLD,
                    EAR_OPEN_THRESHOLD,
                    DROWSY_FRAME_THRESHOLD,
                    HEAD_NOD_FRAMES,
                    EAR_SMOOTHING_ALPHA,
                )

            else:
                self.monitor.update(
                    0.0,
                    False,
                    False,
                    EAR_CLOSED_THRESHOLD,
                    EAR_OPEN_THRESHOLD,
                    DROWSY_FRAME_THRESHOLD,
                    HEAD_NOD_FRAMES,
                    EAR_SMOOTHING_ALPHA,
                )

            if self.monitor.state == DrowsinessState.DROWSY:
                cv2.rectangle(
                    frame,
                    (0, 0),
                    (width, height),
                    (0, 0, 255),
                    8
                )

                alert_text = "DROWSY!"
                text_size = cv2.getTextSize(
                    alert_text,
                    cv2.FONT_HERSHEY_SIMPLEX,
                    2,
                    4
                )[0]

                text_x = (width - text_size[0]) // 2
                text_y = (height + text_size[1]) // 2

                cv2.putText(
                    frame,
                    alert_text,
                    (text_x, text_y),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    2,
                    (0, 0, 255),
                    4
                )

            self._frame_count += 1
            now = time.time()
            if now - self._last_fps_time >= 1.0:
                self.fps = self._frame_count / (now - self._last_fps_time)
                self._frame_count = 0
                self._last_fps_time = now

            with self._lock:
                self._latest_frame = frame.copy()
                self._latest_state = self.monitor.state
                self._latest_ear = self.monitor.smoothed_ear
                self._latest_head_tilt = head_tilt
                self._latest_fps = self.fps

            elapsed = time.time() - loop_start
            target_interval = 1.0 / self.detection_fps
            if elapsed < target_interval:
                time.sleep(target_interval - elapsed)

    def update(self):

        with self._lock:
            if self._latest_frame is None:
                return False, None, DrowsinessState.NO_FACE, 0.0, 0.0, 0

            return (
                True,
                self._latest_frame,
                self._latest_state,
                self._latest_ear,
                self._latest_head_tilt,
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
