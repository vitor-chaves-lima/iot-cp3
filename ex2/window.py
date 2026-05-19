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
    DrowsinessTracker
)

from state import DrowsinessState


class MainWindow(QWidget):

    def __init__(self):

        super().__init__()

        self.tracker = (
            DrowsinessTracker()
        )

        self.setWindowTitle(
            "Drowsiness Detection"
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

            QLabel#alert {
                font-size: 42px;
                font-weight: bold;
                background: transparent;
                color: #00c853;
            }

            QLabel#drowsy {
                font-size: 42px;
                font-weight: bold;
                background: transparent;
                color: #ff1744;
            }

            QLabel#no_face {
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

        state_title = QLabel(
            "STATUS"
        )

        state_title.setObjectName(
            "label"
        )

        sidebar_layout.addWidget(
            state_title
        )

        self.state_label = QLabel(
            "NO_FACE"
        )

        self.state_label.setObjectName(
            "no_face"
        )

        sidebar_layout.addWidget(
            self.state_label
        )

        sidebar_layout.addSpacing(
            32
        )

        separator = QWidget()
        separator.setFixedHeight(1)
        separator.setStyleSheet("background-color: #2a2a2a;")

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

        ear_row = QHBoxLayout()

        ear_label_title = QLabel("EAR")
        ear_label_title.setObjectName("label")

        self.ear_label = QLabel("0.000")
        self.ear_label.setObjectName("debug_value")

        ear_row.addWidget(ear_label_title)
        ear_row.addWidget(self.ear_label, alignment=Qt.AlignRight)
        ear_row.addStretch()

        sidebar_layout.addLayout(ear_row)

        sidebar_layout.addSpacing(
            8
        )

        tilt_row = QHBoxLayout()

        tilt_label_title = QLabel("Head Tilt")
        tilt_label_title.setObjectName("label")

        self.tilt_label = QLabel("0.00")
        self.tilt_label.setObjectName("debug_value")

        tilt_row.addWidget(tilt_label_title)
        tilt_row.addWidget(self.tilt_label, alignment=Qt.AlignRight)
        tilt_row.addStretch()

        sidebar_layout.addLayout(tilt_row)

        sidebar_layout.addSpacing(
            8
        )

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

        det_fps_title = QLabel(
            "Detection FPS"
        )

        det_fps_title.setObjectName(
            "label"
        )

        sidebar_layout.addWidget(
            det_fps_title
        )

        det_fps_row = QHBoxLayout()

        self.btn_det_fps_down = QPushButton("−")
        self.btn_det_fps_down.setFixedWidth(40)
        self.btn_det_fps_down.clicked.connect(
            self.decrease_detection_fps
        )

        self.det_fps_value = QLabel(
            str(self.tracker.detection_fps)
        )

        self.det_fps_value.setObjectName("debug_value")

        self.btn_det_fps_up = QPushButton("+")
        self.btn_det_fps_up.setFixedWidth(40)
        self.btn_det_fps_up.clicked.connect(
            self.increase_detection_fps
        )

        det_fps_row.addWidget(self.btn_det_fps_down)
        det_fps_row.addWidget(self.det_fps_value, alignment=Qt.AlignCenter)
        det_fps_row.addWidget(self.btn_det_fps_up)

        sidebar_layout.addLayout(det_fps_row)

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

        self.btn_eyes = QPushButton(
            "Eyes"
        )

        self.btn_eyes.setCheckable(True)
        self.btn_eyes.setChecked(True)
        self.btn_eyes.clicked.connect(
            self.toggle_eyes
        )

        debug_btns_row.addWidget(
            self.btn_landmarks
        )

        debug_btns_row.addWidget(
            self.btn_eyes
        )

        sidebar_layout.addLayout(
            debug_btns_row
        )

        sidebar_layout.addStretch()

        root_layout.addWidget(
            sidebar
        )

    def toggle_landmarks(self, checked):
        self.tracker.show_landmarks = checked

    def toggle_eyes(self, checked):
        self.tracker.show_eyes = checked

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

    def increase_detection_fps(self):
        self.tracker.increase_detection_fps()
        self.det_fps_value.setText(
            str(self.tracker.detection_fps)
        )

    def decrease_detection_fps(self):
        self.tracker.decrease_detection_fps()
        self.det_fps_value.setText(
            str(self.tracker.detection_fps)
        )

    def update_frame(self):

        (
            success,
            frame,
            state,
            ear,
            head_tilt,
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

        if state == DrowsinessState.ALERT:
            self.state_label.setText("ALERT")
            self.state_label.setObjectName("alert")
        elif state == DrowsinessState.DROWSY:
            self.state_label.setText("DROWSY")
            self.state_label.setObjectName("drowsy")
        else:
            self.state_label.setText("NO_FACE")
            self.state_label.setObjectName("no_face")

        self.state_label.style().unpolish(self.state_label)
        self.state_label.style().polish(self.state_label)

        self.ear_label.setText(
            f"{ear:.3f}"
        )

        self.tilt_label.setText(
            f"{head_tilt:.2f}"
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
