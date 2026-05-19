from pathlib import Path

MODEL_PATH = (
    Path(__file__).resolve().parent
    / "face_landmarker.task"
)

NUM_FACES = 1

MIN_DETECTION_CONFIDENCE = 0.5
MIN_PRESENCE_CONFIDENCE = 0.5
MIN_TRACKING_CONFIDENCE = 0.5

EAR_CLOSED_THRESHOLD = 0.20
EAR_OPEN_THRESHOLD = 0.25

EAR_SMOOTHING_ALPHA = 0.2

DROWSY_FRAME_THRESHOLD = 10

HEAD_TILT_THRESHOLD = 0.35
HEAD_DOWN_THRESHOLD = 0.12

HEAD_NOD_FRAMES = 8

LANDMARK_RADIUS = 3

EYE_CONNECTIONS = [
    (33, 133),
    (160, 159),
    (144, 145),
    (362, 263),
    (387, 386),
    (374, 373),
]

FACE_KEYPOINTS = {
    "left_eye_inner": 33,
    "left_eye_outer": 133,
    "left_eye_upper1": 160,
    "left_eye_upper2": 159,
    "left_eye_lower1": 144,
    "left_eye_lower2": 145,
    "right_eye_inner": 362,
    "right_eye_outer": 263,
    "right_eye_upper1": 387,
    "right_eye_upper2": 386,
    "right_eye_lower1": 374,
    "right_eye_lower2": 373,
    "nose_tip": 1,
    "chin": 152,
    "left_ear": 234,
    "right_ear": 454,
}

DETECTION_RESOLUTIONS = {
    "Low": (224, 168),
    "Medium": (320, 240),
    "High": (480, 360),
}

DETECTION_FPS_OPTIONS = [5, 10, 15, 20, 30]
