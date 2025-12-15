"""
Video player widget with playback controls.
"""
from PySide6.QtWidgets import (QWidget, QVBoxLayout, QHBoxLayout, QLabel,
                                QPushButton, QSlider)
from PySide6.QtCore import Qt, Signal
from PySide6.QtGui import QPixmap, QImage


class VideoPlayerWidget(QWidget):
    """Widget for displaying video and playback controls"""

    play_clicked = Signal()
    pause_clicked = Signal()
    stop_clicked = Signal()
    seek_requested = Signal(int)  # Position in milliseconds
    step_forward_clicked = Signal()
    step_backward_clicked = Signal()
    slider_moved = Signal(int)  # Emitted while dragging slider

    def __init__(self):
        super().__init__()

        self.is_playing = False
        self.duration_ms = 0
        self.total_frames = 0

        self._setup_ui()

    def _setup_ui(self):
        """Setup the user interface"""
        layout = QVBoxLayout(self)

        # Video display area
        self.video_label = QLabel()
        self.video_label.setMinimumSize(640, 480)
        self.video_label.setAlignment(Qt.AlignCenter)
        self.video_label.setStyleSheet("background-color: black; border: 1px solid gray;")
        self.video_label.setText("No video loaded")
        self.video_label.setStyleSheet("background-color: black; color: white; border: 1px solid gray;")
        layout.addWidget(self.video_label, stretch=1)

        # Controls layout
        controls_layout = QHBoxLayout()

        # Step backward button
        self.step_backward_button = QPushButton("◀ Step")
        self.step_backward_button.setEnabled(False)
        self.step_backward_button.setFocusPolicy(Qt.NoFocus)
        self.step_backward_button.clicked.connect(self._on_step_backward_clicked)
        controls_layout.addWidget(self.step_backward_button)

        # Step forward button
        self.step_forward_button = QPushButton("Step ▶")
        self.step_forward_button.setEnabled(False)
        self.step_forward_button.setFocusPolicy(Qt.NoFocus)
        self.step_forward_button.clicked.connect(self._on_step_forward_clicked)
        controls_layout.addWidget(self.step_forward_button)

        # Play button
        self.play_button = QPushButton("Play")
        self.play_button.setEnabled(False)
        self.play_button.setFocusPolicy(Qt.NoFocus)
        self.play_button.clicked.connect(self._on_play_clicked)
        controls_layout.addWidget(self.play_button)

        # Pause button
        self.pause_button = QPushButton("Pause")
        self.pause_button.setEnabled(False)
        self.pause_button.setFocusPolicy(Qt.NoFocus)
        self.pause_button.clicked.connect(self._on_pause_clicked)
        controls_layout.addWidget(self.pause_button)

        # Stop button
        self.stop_button = QPushButton("Stop")
        self.stop_button.setEnabled(False)
        self.stop_button.setFocusPolicy(Qt.NoFocus)
        self.stop_button.clicked.connect(self._on_stop_clicked)
        controls_layout.addWidget(self.stop_button)

        # Info labels layout
        info_layout = QVBoxLayout()

        # Time label
        self.time_label = QLabel("00:00 / 00:00")
        self.time_label.setAlignment(Qt.AlignCenter)
        info_layout.addWidget(self.time_label)

        # Frame label
        self.frame_label = QLabel("Frame: 0 / 0")
        self.frame_label.setAlignment(Qt.AlignCenter)
        info_layout.addWidget(self.frame_label)

        controls_layout.addLayout(info_layout)

        layout.addLayout(controls_layout)

        # Timeline slider
        timeline_layout = QVBoxLayout()
        timeline_label = QLabel("Timeline")
        timeline_label.setAlignment(Qt.AlignCenter)
        timeline_layout.addWidget(timeline_label)

        self.timeline_slider = QSlider(Qt.Horizontal)
        self.timeline_slider.setEnabled(False)
        self.timeline_slider.setMinimum(0)
        self.timeline_slider.setMaximum(100)
        self.timeline_slider.setValue(0)
        self.timeline_slider.sliderPressed.connect(self._on_slider_pressed)
        self.timeline_slider.sliderReleased.connect(self._on_slider_released)
        self.timeline_slider.sliderMoved.connect(self._on_slider_moved)
        timeline_layout.addWidget(self.timeline_slider)

        layout.addLayout(timeline_layout)

        self.slider_pressed = False

    def _on_play_clicked(self):
        """Handle play button click"""
        self.is_playing = True
        self.play_button.setEnabled(False)
        self.pause_button.setEnabled(True)
        self.stop_button.setEnabled(True)
        self.play_clicked.emit()

    def _on_pause_clicked(self):
        """Handle pause button click"""
        self.is_playing = False
        self.play_button.setEnabled(True)
        self.pause_button.setEnabled(False)
        self.pause_clicked.emit()

    def _on_stop_clicked(self):
        """Handle stop button click"""
        self.is_playing = False
        self.play_button.setEnabled(True)
        self.pause_button.setEnabled(False)
        self.stop_button.setEnabled(False)
        self.timeline_slider.setValue(0)
        self.time_label.setText("00:00 / 00:00")
        self.stop_clicked.emit()

    def _on_step_forward_clicked(self):
        """Handle step forward button click"""
        self.step_forward_clicked.emit()

    def _on_step_backward_clicked(self):
        """Handle step backward button click"""
        self.step_backward_clicked.emit()

    def _on_slider_pressed(self):
        """Handle slider press"""
        self.slider_pressed = True

    def _on_slider_moved(self, position_ms):
        """Handle slider being moved (while dragging)"""
        # Update time label while dragging
        current_time = self._format_time(position_ms)
        total_time = self._format_time(self.duration_ms)
        self.time_label.setText(f"{current_time} / {total_time}")

        # Emit signal to update video frame preview
        self.slider_moved.emit(position_ms)

    def _on_slider_released(self):
        """Handle slider release"""
        self.slider_pressed = False
        position_ms = self.timeline_slider.value()
        self.seek_requested.emit(position_ms)

    def display_frame(self, image: QImage):
        """
        Display a video frame

        Args:
            image: QImage to display
        """
        if image:
            pixmap = QPixmap.fromImage(image)
            # Scale to fit while maintaining aspect ratio
            scaled_pixmap = pixmap.scaled(
                self.video_label.size(),
                Qt.KeepAspectRatio,
                Qt.SmoothTransformation
            )
            self.video_label.setPixmap(scaled_pixmap)

    def update_position(self, position_ms: int, current_frame: int = 0):
        """
        Update the timeline position

        Args:
            position_ms: Current position in milliseconds
            current_frame: Current frame number
        """
        if not self.slider_pressed:
            self.timeline_slider.setValue(position_ms)

        # Update time label
        current_time = self._format_time(position_ms)
        total_time = self._format_time(self.duration_ms)
        self.time_label.setText(f"{current_time} / {total_time}")

        # Update frame label
        self.frame_label.setText(f"Frame: {current_frame} / {self.total_frames}")

    def set_duration(self, duration_ms: int, total_frames: int = 0):
        """
        Set the video duration

        Args:
            duration_ms: Duration in milliseconds
            total_frames: Total number of frames
        """
        self.duration_ms = duration_ms
        self.total_frames = total_frames
        self.timeline_slider.setMaximum(duration_ms)
        self.timeline_slider.setEnabled(True)
        self.frame_label.setText(f"Frame: 0 / {total_frames}")

    def enable_controls(self, enabled: bool):
        """Enable or disable playback controls"""
        self.play_button.setEnabled(enabled and not self.is_playing)
        self.pause_button.setEnabled(enabled and self.is_playing)
        self.stop_button.setEnabled(enabled)
        self.step_forward_button.setEnabled(enabled)
        self.step_backward_button.setEnabled(enabled)
        self.timeline_slider.setEnabled(enabled)

    def reset(self):
        """Reset the player to initial state"""
        self.is_playing = False
        self.duration_ms = 0
        self.total_frames = 0
        self.video_label.clear()
        self.video_label.setText("No video loaded")
        self.timeline_slider.setValue(0)
        self.timeline_slider.setEnabled(False)
        self.time_label.setText("00:00 / 00:00")
        self.frame_label.setText("Frame: 0 / 0")
        self.play_button.setEnabled(False)
        self.pause_button.setEnabled(False)
        self.stop_button.setEnabled(False)
        self.step_forward_button.setEnabled(False)
        self.step_backward_button.setEnabled(False)

    def on_playback_finished(self):
        """Handle playback finished"""
        self.is_playing = False
        self.play_button.setEnabled(True)
        self.pause_button.setEnabled(False)
        self.stop_button.setEnabled(False)

    @staticmethod
    def _format_time(ms: int) -> str:
        """
        Format milliseconds to MM:SS

        Args:
            ms: Time in milliseconds

        Returns:
            Formatted time string
        """
        total_seconds = ms // 1000
        minutes = total_seconds // 60
        seconds = total_seconds % 60
        return f"{minutes:02d}:{seconds:02d}"
