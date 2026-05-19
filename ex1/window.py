# src/window.py

import time

import cv2

from PySide6.QtCore import Qt, QTimer
from PySide6.QtGui import (
    QImage,
    QPixmap
)
from PySide6.QtWidgets import (
    QWidget,
    QLabel,
    QHBoxLayout,
    QVBoxLayout,
    QApplication,
    QPushButton,
)

from tracker import (
    JumpingJackTracker
)


class MainWindow(QWidget):

    def __init__(self):

        super().__init__()

        self.tracker = (
            JumpingJackTracker()
        )

        self.setWindowTitle(
            "Jumping Jack Tracker"
        )

        self.resize(1400, 800)

        self.setStyleSheet("""
            QWidget {
                background-color: #121212;
                color: white;
            }

            QLabel {
                background: transparent;
            }

            QLabel#title {
                font-size: 26px;
                font-weight: bold;
                background: transparent;
            }

            QLabel#label {
                color: #888;
                font-size: 12px;
                letter-spacing: 1px;
                background: transparent;
            }

            QLabel#value {
                font-size: 42px;
                font-weight: bold;
                background: transparent;
            }

            QLabel#exercising {
                font-size: 42px;
                font-weight: bold;
                background: transparent;
                color: #00c853;
            }

            QLabel#idle {
                font-size: 42px;
                font-weight: bold;
                background: transparent;
                color: #888;
            }

            QLabel#debug_value {
                font-size: 18px;
                font-weight: bold;
                background: transparent;
                color: #aaa;
            }

            QWidget#sidebar {
                background-color: #1c1c1c;
                border-left: 1px solid #2a2a2a;
            }

            QPushButton {
                background-color: #2a2a2a;
                border: 1px solid #3a3a3a;
                border-radius: 6px;
                padding: 8px 16px;
                font-size: 13px;
                color: white;
            }

            QPushButton:hover {
                background-color: #3a3a3a;
            }

            QPushButton:checked {
                background-color: #0078d4;
                border-color: #0078d4;
            }

            QPushButton#reset {
                background-color: #8b0000;
                border-color: #a00000;
            }

            QPushButton#reset:hover {
                background-color: #a00000;
            }
        """)

        self.build_ui()

        self.timer = QTimer()

        self.timer.timeout.connect(
            self.update_frame
        )

        self.timer.start(30)

        self._ui_fps = 0
        self._ui_frame_count = 0
        self._ui_last_fps_time = time.time()

    def build_ui(self):

        root_layout = (
            QHBoxLayout(self)
        )

        root_layout.setContentsMargins(
            0,
            0,
            0,
            0
        )

        root_layout.setSpacing(0)

        self.video_label = QLabel()

        self.video_label.setAlignment(
            Qt.AlignCenter
        )

        root_layout.addWidget(
            self.video_label,
            stretch=1
        )

        sidebar = QWidget()

        sidebar.setObjectName(
            "sidebar"
        )

        sidebar.setFixedWidth(320)

        sidebar_layout = (
            QVBoxLayout(sidebar)
        )

        sidebar_layout.setContentsMargins(
            32,
            32,
            32,
            32
        )

        sidebar_layout.setSpacing(
            18
        )

        title = QLabel(
            "FITNESS TRACKER"
        )

        title.setObjectName(
            "title"
        )

        sidebar_layout.addWidget(
            title
        )

        sidebar_layout.addSpacing(
            24
        )

        exercise_title = QLabel(
            "EXERCISE"
        )

        exercise_title.setObjectName(
            "label"
        )

        sidebar_layout.addWidget(
            exercise_title
        )

        self.exercise_label = QLabel(
            "IDLE"
        )

        self.exercise_label.setObjectName(
            "idle"
        )

        sidebar_layout.addWidget(
            self.exercise_label
        )

        sidebar_layout.addSpacing(
            32
        )

        counter_title = QLabel(
            "TOTAL COUNT"
        )

        counter_title.setObjectName(
            "label"
        )

        sidebar_layout.addWidget(
            counter_title
        )

        counter_row = QHBoxLayout()

        self.counter_label = QLabel(
            "0"
        )

        self.counter_label.setObjectName(
            "value"
        )

        self.btn_reset = QPushButton(
            "Reset"
        )

        self.btn_reset.setObjectName("reset")
        self.btn_reset.setFixedWidth(80)
        self.btn_reset.clicked.connect(
            self.reset_counter
        )

        counter_row.addWidget(
            self.counter_label
        )

        counter_row.addWidget(
            self.btn_reset,
            alignment=Qt.AlignRight
        )

        sidebar_layout.addLayout(
            counter_row
        )

        sidebar_layout.addSpacing(
            32
        )

        separator = QWidget()
        separator.setObjectName("debug_separator")
        separator.setFixedHeight(1)

        sidebar_layout.addWidget(
            separator
        )

        sidebar_layout.addSpacing(
            16
        )

        debug_title = QLabel(
            "DEBUG"
        )

        debug_title.setObjectName(
            "label"
        )

        sidebar_layout.addWidget(
            debug_title
        )

        det_row = QHBoxLayout()

        det_label_title = QLabel("Detection")
        det_label_title.setObjectName("label")

        self.det_label = QLabel("CLOSED")
        self.det_label.setObjectName("debug_value")

        det_row.addWidget(det_label_title)
        det_row.addWidget(self.det_label, alignment=Qt.AlignRight)
        det_row.addStretch()

        sidebar_layout.addLayout(det_row)

        fps_row = QHBoxLayout()

        fps_label_title = QLabel("FPS")
        fps_label_title.setObjectName("label")

        self.fps_label = QLabel("0")
        self.fps_label.setObjectName("debug_value")

        fps_row.addWidget(fps_label_title)
        fps_row.addWidget(self.fps_label, alignment=Qt.AlignRight)
        fps_row.addStretch()

        sidebar_layout.addLayout(fps_row)

        sidebar_layout.addSpacing(
            12
        )

        det_res_title = QLabel(
            "Detection Resolution"
        )

        det_res_title.setObjectName(
            "label"
        )

        sidebar_layout.addWidget(
            det_res_title
        )

        det_res_row = QHBoxLayout()

        self.btn_det_res_down = QPushButton("−")
        self.btn_det_res_down.setFixedWidth(40)
        self.btn_det_res_down.clicked.connect(
            self.decrease_detection_resolution
        )

        self.det_res_value = QLabel(
            self.tracker.detection_resolution_label
        )

        self.det_res_value.setObjectName("debug_value")

        self.btn_det_res_up = QPushButton("+")
        self.btn_det_res_up.setFixedWidth(40)
        self.btn_det_res_up.clicked.connect(
            self.increase_detection_resolution
        )

        det_res_row.addWidget(self.btn_det_res_down)
        det_res_row.addWidget(self.det_res_value, alignment=Qt.AlignCenter)
        det_res_row.addWidget(self.btn_det_res_up)

        sidebar_layout.addLayout(det_res_row)

        sidebar_layout.addSpacing(
            12
        )

        debug_btns_row = QHBoxLayout()

        self.btn_landmarks = QPushButton(
            "Landmarks"
        )

        self.btn_landmarks.setCheckable(True)
        self.btn_landmarks.setChecked(True)
        self.btn_landmarks.clicked.connect(
            self.toggle_landmarks
        )

        self.btn_skeleton = QPushButton(
            "Skeleton"
        )

        self.btn_skeleton.setCheckable(True)
        self.btn_skeleton.setChecked(True)
        self.btn_skeleton.clicked.connect(
            self.toggle_skeleton
        )

        debug_btns_row.addWidget(
            self.btn_landmarks
        )

        debug_btns_row.addWidget(
            self.btn_skeleton
        )

        sidebar_layout.addLayout(
            debug_btns_row
        )

        sidebar_layout.addStretch()

        root_layout.addWidget(
            sidebar
        )

    def reset_counter(self):
        self.tracker.state.reset()

    def toggle_landmarks(self, checked):
        self.tracker.show_landmarks = checked

    def toggle_skeleton(self, checked):
        self.tracker.show_skeleton = checked

    def increase_detection_resolution(self):
        self.tracker.increase_detection_resolution()
        self.det_res_value.setText(
            self.tracker.detection_resolution_label
        )

    def decrease_detection_resolution(self):
        self.tracker.decrease_detection_resolution()
        self.det_res_value.setText(
            self.tracker.detection_resolution_label
        )

    def update_frame(self):

        (
            success,
            frame,
            detection,
            counting,
            counter,
            fps,
        ) = self.tracker.update()

        if not success:
            return

        frame = cv2.cvtColor(
            frame,
            cv2.COLOR_BGR2RGB
        )

        height, width, channel = (
            frame.shape
        )

        bytes_per_line = (
            channel * width
        )

        image = QImage(
            frame.data,
            width,
            height,
            bytes_per_line,
            QImage.Format_RGB888
        )

        pixmap = (
            QPixmap.fromImage(
                image
            )
        )

        self.video_label.setPixmap(
            pixmap.scaled(
                self.video_label.size(),
                Qt.KeepAspectRatio,
                Qt.SmoothTransformation
            )
        )

        if counting:
            self.exercise_label.setText("EXERCISING")
            self.exercise_label.setObjectName("exercising")
        else:
            self.exercise_label.setText("IDLE")
            self.exercise_label.setObjectName("idle")

        self.exercise_label.style().unpolish(self.exercise_label)
        self.exercise_label.style().polish(self.exercise_label)

        self.det_label.setText(detection.value.upper())

        self.counter_label.setText(
            str(counter)
        )

        self.fps_label.setText(
            f"{fps:.1f}"
        )

        self._ui_frame_count += 1
        now = time.time()
        if now - self._ui_last_fps_time >= 1.0:
            self._ui_fps = self._ui_frame_count / (now - self._ui_last_fps_time)
            self._ui_frame_count = 0
            self._ui_last_fps_time = now

    def closeEvent(
        self,
        event
    ):

        self.timer.stop()

        self.tracker.release()

        event.accept()
