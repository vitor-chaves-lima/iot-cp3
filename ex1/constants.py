from pathlib import Path

MODEL_PATH = (
    Path(__file__).resolve().parent
    / "pose_landmarker_full.task"
)

NUM_POSES = 1

MIN_DETECTION_CONFIDENCE = 0.5
MIN_PRESENCE_CONFIDENCE = 0.5
MIN_TRACKING_CONFIDENCE = 0.5

DEBOUNCE_FRAMES = 5

STATE_TIMEOUT = 0.75

IDLE_TIMEOUT = 2.0

LEGS_OPEN_FACTOR = 2.0
LEGS_CLOSED_FACTOR = 1.2

LANDMARK_RADIUS = 4

SKELETON_CONNECTIONS = [
    (11, 12), (11, 13), (13, 15),
    (12, 14), (14, 16),
    (11, 23), (12, 24),
    (23, 24), (23, 25), (25, 27),
    (24, 26), (26, 28),
]

DETECTION_RESOLUTIONS = {
    "Low": (224, 168),
    "Medium": (320, 240),
    "High": (480, 360),
}

DETECTION_FPS_OPTIONS = [5, 10, 15, 20, 30]
