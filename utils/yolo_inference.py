"""
YOLO model inference utilities.
"""
import numpy as np
from typing import Optional


class YOLOInference:
    """Handles YOLO model loading and inference"""

    def __init__(self):
        self.model = None
        self.confidence = 0.5
        self.enabled = True
        self.model_path = None

    def load_model(self, path: str) -> bool:
        """
        Load a YOLO model from .pt file

        Args:
            path: Path to .pt model file

        Returns:
            True if loaded successfully, False otherwise
        """
        try:
            from ultralytics import YOLO
            self.model = YOLO(path)
            self.model_path = path
            return True
        except Exception as e:
            print(f"Error loading model: {e}")
            self.model = None
            self.model_path = None
            self.model_path = None
            return False

    def unload_model(self):
        """Unload the current model"""
        self.model = None
        self.model_path = None

    def predict(self, frame: np.ndarray):
        """
        Run inference on a frame

        Args:
            frame: Input frame (numpy array in BGR format from OpenCV)

        Returns:
            Results object from ultralytics, or None if model not loaded
        """
        if self.model is None or not self.enabled:
            return None

        try:
            results = self.model(frame, conf=self.confidence, verbose=False)
            return results[0] if results else None
        except Exception as e:
            print(f"Error during inference: {e}")
            return None

    def draw_results(self, frame: np.ndarray, results) -> np.ndarray:
        """
        Draw bounding boxes, labels, confidence scores, and masks on frame

        Args:
            frame: Input frame
            results: Results object from predict()

        Returns:
            Annotated frame
        """
        if results is None:
            return frame

        try:
            # Use ultralytics built-in plotting which includes boxes, labels, scores, and masks
            annotated_frame = results.plot()
            return annotated_frame
        except Exception as e:
            print(f"Error drawing results: {e}")
            return frame

    def set_confidence(self, value: float):
        """
        Set confidence threshold

        Args:
            value: Confidence threshold (0.0 to 1.0)
        """
        self.confidence = max(0.0, min(1.0, value))

    def set_enabled(self, enabled: bool):
        """
        Enable or disable inference

        Args:
            enabled: True to enable, False to disable
        """
        self.enabled = enabled

    def is_loaded(self) -> bool:
        """Check if a model is loaded"""
        return self.model is not None

    def get_model_path(self) -> Optional[str]:
        """Get the path of the loaded model"""
        return self.model_path
