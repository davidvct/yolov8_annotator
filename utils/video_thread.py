"""
Video processing thread for smooth playback and inference.
"""
import cv2
import numpy as np
from PySide6.QtCore import QThread, Signal
from PySide6.QtGui import QImage
from utils.yolo_inference import YOLOInference


class VideoThread(QThread):
    """Thread for processing video frames"""

    frame_ready = Signal(QImage)        # Emit processed frame
    position_changed = Signal(int, int) # Emit current position (ms) and current frame number
    duration_changed = Signal(int, int) # Emit total duration (ms) and total frames
    playback_finished = Signal()        # Emit when video ends
    error_occurred = Signal(str)        # Emit error message
    
    # Signals for frame export
    request_current_frame_data = Signal()
    # frame_data_ready_for_export sends (frame_np, frame_number, video_name, results)
    frame_data_ready_for_export = Signal(object, int, str, object)

    def __init__(self):
        super().__init__()

        self.video_path = None
        self.inference_engine = None
        self.is_playing = False
        self.is_paused = False
        self.should_stop = False
        self.seek_position = -1
        self.cap = None
        self.fps = 0
        self.current_frame_number = 0  # Track current frame manually
        
        # Connect internal signal to handler slot running in this thread context
        self.request_current_frame_data.connect(self._export_frame_data_handler)

    def set_video(self, video_path: str):
        """Set the video file to play"""
        self.video_path = video_path
        self.should_stop = True  # Stop current playback

    def set_inference_engine(self, engine: YOLOInference):
        """Set the YOLO inference engine"""
        self.inference_engine = engine

    def _export_frame_data_handler(self):
        """
        Retrieves current frame data and inference results, and emits them for export.
        This method is executed in the VideoThread when requested via signal.
        """
        if not self.cap or not self.cap.isOpened():
            self.error_occurred.emit("Cannot export: video stream is not open.")
            return

        if not self.is_paused:
            # Should be paused, but defensively check
            return

        # 1. Read the currently tracked frame
        # We use _read_frame_at to ensure we read the specific frame number we are currently on.
        frame = self._read_frame_at(self.current_frame_number)

        if frame is None:
            self.error_occurred.emit(f"Cannot export: failed to read frame {self.current_frame_number}.")
            return

        # 2. Run inference if enabled
        results = None
        if self.inference_engine and self.inference_engine.is_loaded() and self.inference_engine.enabled:
            # Predict uses the raw frame (non-annotated)
            results = self.inference_engine.predict(frame)
        
        # 3. Get video name
        video_name = cv2.os.path.splitext(cv2.os.path.basename(self.video_path))[0]
        
        # 4. Emit data
        # frame_np is the raw frame (BGR, no drawn annotations)
        self.frame_data_ready_for_export.emit(frame, self.current_frame_number, video_name, results)


    def run(self):
        """Main thread loop"""
        if not self.video_path:
            return

        # Open video
        self.cap = cv2.VideoCapture(self.video_path)

        if not self.cap.isOpened():
            self.error_occurred.emit(f"Failed to open video: {self.video_path}")
            return

        # Get video properties
        self.fps = self.cap.get(cv2.CAP_PROP_FPS)
        total_frames = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT))
        duration_ms = int((total_frames / self.fps) * 1000) if self.fps > 0 else 0

        self.duration_changed.emit(duration_ms, total_frames)

        self.should_stop = False
        self.is_playing = True
        self.current_frame_number = 0

        while not self.should_stop:
            # Handle seek
            if self.seek_position >= 0:
                self.cap.set(cv2.CAP_PROP_POS_MSEC, self.seek_position)
                self.current_frame_number = int(self.cap.get(cv2.CAP_PROP_POS_FRAMES))
                self.seek_position = -1

            # Handle pause
            if self.is_paused:
                self.msleep(100)
                continue

            # Read frame
            ret, frame = self.cap.read()

            if not ret:
                # End of video
                self.playback_finished.emit()
                break

            # Run inference if enabled
            if self.inference_engine and self.inference_engine.is_loaded() and self.inference_engine.enabled:
                results = self.inference_engine.predict(frame)
                frame = self.inference_engine.draw_results(frame, results)

            # Convert frame to QImage
            qt_image = self._convert_frame_to_qimage(frame)
            if qt_image:
                self.frame_ready.emit(qt_image)

            # Update position
            position_ms = int(self.cap.get(cv2.CAP_PROP_POS_MSEC))
            
            # Update frame index directly from the capture device to ensure it matches the frame we just read
            # CAP_PROP_POS_FRAMES returns the index of the next frame to be captured, 
            # so the current frame we hold is one less.
            self.current_frame_number = int(self.cap.get(cv2.CAP_PROP_POS_FRAMES)) - 1
            
            self.position_changed.emit(position_ms, self.current_frame_number)

            # Control playback speed (basic timing)
            if self.fps > 0:
                delay_ms = int(1000 / self.fps)
                self.msleep(delay_ms)

        # Cleanup
        if self.cap:
            self.cap.release()
            self.cap = None
        self.is_playing = False

    def play(self):
        """Start or resume playback"""
        if not self.is_playing:
            self.start()
        else:
            self.is_paused = False

    def pause(self):
        """Pause playback"""
        self.is_paused = True

    def stop(self):
        """Stop playback"""
        self.should_stop = True
        self.is_paused = False
        if self.isRunning():
            self.wait()

    def seek(self, position_ms: int):
        """
        Seek to a specific position

        Args:
            position_ms: Position in milliseconds
        """
        self.seek_position = position_ms

    def step_forward(self):
        """Step one frame forward"""
        if not self.cap or not self.cap.isOpened():
            return

        # Pause playback
        self.is_paused = True

        # Calculate next frame number
        total_frames = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT))
        next_frame = self.current_frame_number + 1

        # Check bounds
        if next_frame >= total_frames:
            return

        # Read next frame without advancing position
        frame = self._read_frame_at(next_frame)
        if frame is not None:
            # Update frame counter
            self.current_frame_number = next_frame

            # Run inference if enabled
            if self.inference_engine and self.inference_engine.is_loaded() and self.inference_engine.enabled:
                results = self.inference_engine.predict(frame)
                frame = self.inference_engine.draw_results(frame, results)

            # Convert and emit frame
            qt_image = self._convert_frame_to_qimage(frame)
            if qt_image:
                self.frame_ready.emit(qt_image)

            # Update position
            position_ms = int(self.cap.get(cv2.CAP_PROP_POS_MSEC))
            self.position_changed.emit(position_ms, self.current_frame_number)

    def step_backward(self):
        """Step one frame backward"""
        if not self.cap or not self.cap.isOpened():
            return

        # Pause playback
        self.is_paused = True

        # Can't go back from frame 0
        if self.current_frame_number <= 0:
            return

        # Calculate previous frame number
        prev_frame = self.current_frame_number - 1

        # Read previous frame without advancing position
        frame = self._read_frame_at(prev_frame)
        if frame is not None:
            # Update frame counter
            self.current_frame_number = prev_frame

            # Run inference if enabled
            if self.inference_engine and self.inference_engine.is_loaded() and self.inference_engine.enabled:
                results = self.inference_engine.predict(frame)
                frame = self.inference_engine.draw_results(frame, results)

            # Convert and emit frame
            qt_image = self._convert_frame_to_qimage(frame)
            if qt_image:
                self.frame_ready.emit(qt_image)

            # Update position
            position_ms = int(self.cap.get(cv2.CAP_PROP_POS_MSEC))
            self.position_changed.emit(position_ms, self.current_frame_number)

    def get_frame_at_position(self, position_ms: int):
        """
        Get a frame at a specific position (for slider preview)

        Args:
            position_ms: Position in milliseconds
        """
        if not self.cap or not self.cap.isOpened():
            return

        # Calculate frame number from time position and FPS for accuracy
        if self.fps > 0:
            # Convert milliseconds to seconds, then to frame number
            frame_number = int((position_ms / 1000.0) * self.fps)
            # Ensure we don't exceed total frames
            total_frames = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT))
            frame_number = min(frame_number, total_frames - 1)
        else:
            # Fallback to time-based seek
            self.cap.set(cv2.CAP_PROP_POS_MSEC, position_ms)
            frame_number = int(self.cap.get(cv2.CAP_PROP_POS_FRAMES))

        # Read frame without advancing position
        frame = self._read_frame_at(frame_number)
        if frame is not None:
            # Run inference if enabled
            if self.inference_engine and self.inference_engine.is_loaded() and self.inference_engine.enabled:
                results = self.inference_engine.predict(frame)
                frame = self.inference_engine.draw_results(frame, results)

            # Convert and emit frame
            qt_image = self._convert_frame_to_qimage(frame)
            if qt_image:
                self.frame_ready.emit(qt_image)

            # Update position with frame number
            position_ms = int(self.cap.get(cv2.CAP_PROP_POS_MSEC))
            self.position_changed.emit(position_ms, frame_number)

            # Update our tracked frame number if paused (for subsequent steps)
            if self.is_paused:
                self.current_frame_number = frame_number

    def _read_frame_at(self, frame_number: int):
        """
        Read a specific frame without advancing the position.

        Args:
            frame_number: Frame number to read (0-indexed)

        Returns:
            Frame as numpy array, or None if read failed
        """
        self.cap.set(cv2.CAP_PROP_POS_FRAMES, frame_number)
        ret, frame = self.cap.read()
        if ret:
            # Reset position back to the frame we just read
            # so subsequent operations start from correct position
            self.cap.set(cv2.CAP_PROP_POS_FRAMES, frame_number)
            return frame
        return None

    def _convert_frame_to_qimage(self, frame: np.ndarray) -> QImage:
        """
        Convert OpenCV frame (BGR) to QImage

        Args:
            frame: OpenCV frame in BGR format

        Returns:
            QImage or None if conversion fails
        """
        try:
            # Convert BGR to RGB
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            h, w, ch = rgb_frame.shape
            bytes_per_line = ch * w
            qt_image = QImage(rgb_frame.data, w, h, bytes_per_line, QImage.Format_RGB888)
            # Make a copy to avoid issues with the data going out of scope
            return qt_image.copy()
        except Exception as e:
            print(f"Error converting frame: {e}")
            return None
