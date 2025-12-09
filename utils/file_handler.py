"""
File handling utilities for managing images and annotations.
"""
import os
from typing import List, Tuple
from pathlib import Path


class FileHandler:
    """Handles file operations for images and annotations"""

    # Supported image formats
    IMAGE_EXTENSIONS = {'.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.tif'}

    def __init__(self, images_dir: str = None, labels_dir: str = None):
        """
        Args:
            images_dir: Directory containing images
            labels_dir: Directory containing label files
        """
        self.images_dir = images_dir
        self.labels_dir = labels_dir
        self.image_files: List[str] = []
        self.current_index = 0

        if images_dir:
            self.load_image_list()

    def set_directories(self, images_dir: str, labels_dir: str) -> None:
        """Set the images and labels directories"""
        self.images_dir = images_dir
        self.labels_dir = labels_dir
        self.load_image_list()

    def load_image_list(self) -> None:
        """Load list of image files from the images directory"""
        if not self.images_dir or not os.path.exists(self.images_dir):
            self.image_files = []
            self.current_index = 0
            return

        # Fast method: just check extension, don't check if it's a file
        # This is much faster with thousands of files
        self.image_files = [
            file for file in sorted(os.listdir(self.images_dir))
            if Path(file).suffix.lower() in self.IMAGE_EXTENSIONS
        ]

        self.current_index = 0 if self.image_files else 0

    def get_current_image_path(self) -> str:
        """Get the path to the current image"""
        if not self.image_files:
            return ""
        return os.path.join(self.images_dir, self.image_files[self.current_index])

    def get_current_label_path(self) -> str:
        """Get the path to the current label file"""
        if not self.image_files or not self.labels_dir:
            return ""

        image_name = Path(self.image_files[self.current_index]).stem
        return os.path.join(self.labels_dir, f'{image_name}.txt')

    def next_image(self) -> bool:
        """
        Move to the next image.

        Returns:
            True if successfully moved, False if at the end
        """
        if not self.image_files:
            return False

        if self.current_index < len(self.image_files) - 1:
            self.current_index += 1
            return True
        return False

    def previous_image(self) -> bool:
        """
        Move to the previous image.

        Returns:
            True if successfully moved, False if at the beginning
        """
        if not self.image_files:
            return False

        if self.current_index > 0:
            self.current_index -= 1
            return True
        return False

    def get_current_index(self) -> int:
        """Get the current image index (0-based)"""
        return self.current_index

    def get_total_images(self) -> int:
        """Get the total number of images"""
        return len(self.image_files)

    def get_current_image_name(self) -> str:
        """Get the name of the current image file"""
        if not self.image_files:
            return ""
        return self.image_files[self.current_index]

    def has_images(self) -> bool:
        """Check if there are any images loaded"""
        return len(self.image_files) > 0

    def get_progress_string(self) -> str:
        """Get a string representing current progress (e.g., '1/10')"""
        if not self.image_files:
            return "0/0"
        return f"{self.current_index + 1}/{len(self.image_files)}"

    def goto_image(self, index: int) -> bool:
        """
        Go to a specific image by index.

        Args:
            index: 0-based index

        Returns:
            True if successfully moved, False if index out of range
        """
        if 0 <= index < len(self.image_files):
            self.current_index = index
            return True
        return False
