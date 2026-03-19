"""
YOLO v8 format utilities for parsing and saving annotations.
YOLO format: class_id x1 y1 x2 y2 ... xn yn (normalized coordinates 0-1)
"""
import os
from typing import List, Tuple
from pathlib import Path


class YOLOAnnotation:
    """Represents a single YOLO annotation (polygon or bounding box)"""

    def __init__(self, class_id: int, points: List[Tuple[float, float]]):
        """
        Args:
            class_id: Integer class ID
            points: List of (x, y) tuples in normalized coordinates (0-1)
                    For segmentation, this is a list of polygon vertices.
                    For detection, this should be a list of 4 points representing the corners of the box.
        """
        self.class_id = class_id
        self.points = points  # Normalized coordinates

    def to_yolo_string(self, mode: str = "segmentation") -> str:
        """Convert annotation to YOLO format string"""
        if mode == "detection":
            # Find min and max x/y
            x_coords = [p[0] for p in self.points]
            y_coords = [p[1] for p in self.points]
            min_x, max_x = min(x_coords), max(x_coords)
            min_y, max_y = min(y_coords), max(y_coords)
            
            x_center = (min_x + max_x) / 2.0
            y_center = (min_y + max_y) / 2.0
            width = max_x - min_x
            height = max_y - min_y
            
            return f'{self.class_id} {x_center:.6f} {y_center:.6f} {width:.6f} {height:.6f}'
        else:
            # Segmentation
            coords = ' '.join(f'{x:.6f} {y:.6f}' for x, y in self.points)
            return f'{self.class_id} {coords}'

    @staticmethod
    def from_yolo_string(line: str, mode: str = "segmentation") -> 'YOLOAnnotation':
        """Parse YOLO format string to annotation"""
        parts = line.strip().split()
        if len(parts) < 3:  # At least class_id + 1 point (x, y) / bounding box
            raise ValueError(f"Invalid YOLO format: {line}")

        class_id = int(parts[0])
        coords = [float(x) for x in parts[1:]]

        if mode == "detection":
            if len(coords) != 4:
                raise ValueError(f"Expected 4 values for detection, got {len(coords)}: {line}")
            x_center, y_center, width, height = coords
            # Convert to 4 points polygon
            half_w = width / 2.0
            half_h = height / 2.0
            points = [
                (x_center - half_w, y_center - half_h), # Top-left
                (x_center + half_w, y_center - half_h), # Top-right
                (x_center + half_w, y_center + half_h), # Bottom-right
                (x_center - half_w, y_center + half_h)  # Bottom-left
            ]
        else:
            if len(coords) % 2 != 0:
                raise ValueError(f"Odd number of coordinates for segmentation: {line}")
            points = [(coords[i], coords[i + 1]) for i in range(0, len(coords), 2)]
            
        return YOLOAnnotation(class_id, points)

    def to_pixel_coords(self, img_width: int, img_height: int) -> List[Tuple[float, float]]:
        """Convert normalized coordinates to pixel coordinates"""
        return [(x * img_width, y * img_height) for x, y in self.points]

    @staticmethod
    def from_pixel_coords(class_id: int, pixel_points: List[Tuple[float, float]],
                         img_width: int, img_height: int) -> 'YOLOAnnotation':
        """Create annotation from pixel coordinates"""
        normalized_points = [(x / img_width, y / img_height) for x, y in pixel_points]
        return YOLOAnnotation(class_id, normalized_points)


def load_annotations(annotation_path: str, mode: str = "segmentation") -> List[YOLOAnnotation]:
    """
    Load annotations from a YOLO format file.

    Args:
        annotation_path: Path to the annotation txt file
        mode: "segmentation" or "detection"

    Returns:
        List of YOLOAnnotation objects
    """
    if not os.path.exists(annotation_path):
        return []

    annotations = []
    with open(annotation_path, 'r') as f:
        for line in f:
            line = line.strip()
            if line:  # Skip empty lines
                try:
                    annotation = YOLOAnnotation.from_yolo_string(line, mode=mode)
                    annotations.append(annotation)
                except ValueError as e:
                    print(f"Warning: Skipping invalid line in {annotation_path}: {e}")

    return annotations


def save_annotations(annotation_path: str, annotations: List[YOLOAnnotation], mode: str = "segmentation") -> None:
    """
    Save annotations to a YOLO format file.

    Args:
        annotation_path: Path to save the annotation txt file
        annotations: List of YOLOAnnotation objects
        mode: "segmentation" or "detection"
    """
    if not annotations:
        # Delete the file if no annotations
        if os.path.exists(annotation_path):
            os.remove(annotation_path)
        return

    # Create directory if it doesn't exist
    os.makedirs(os.path.dirname(annotation_path), exist_ok=True)

    with open(annotation_path, 'w') as f:
        for annotation in annotations:
            f.write(annotation.to_yolo_string(mode=mode) + '\n')


def get_annotation_path(image_path: str, labels_dir: str) -> str:
    """
    Get the corresponding annotation file path for an image.

    Args:
        image_path: Path to the image file
        labels_dir: Directory containing label files

    Returns:
        Path to the annotation txt file
    """
    image_name = Path(image_path).stem  # Filename without extension
    return os.path.join(labels_dir, f'{image_name}.txt')


def convert_results_to_yolo_strings(results) -> List[str]:
    """
    Convert ultralytics inference results (segmentation masks) to YOLO format strings.

    Note: This requires 'ultralytics' to be installed and assumes a segmentation model.
    It expects the 'results' object to have a 'masks' attribute.

    Args:
        results: An ultralytics Results object (e.g., results[0] from model())

    Returns:
        List of YOLO annotation strings (class_id x1 y1 x2 y2 ...)
    """
    yolo_strings = []

    # Check if segmentation masks are present (YOLO segmentation model)
    if results and hasattr(results, 'masks') and results.masks is not None:
        try:
            # normalized_polygons is a list of numpy arrays, each array is shape (N, 2)
            normalized_polygons = results.masks.xyn
            # Get class IDs from boxes results (must match the order of masks)
            class_ids = results.boxes.cls.int().tolist()

            if len(normalized_polygons) != len(class_ids):
                print("Warning: Mismatch between number of masks and class IDs during conversion.")
                return []

            for class_id, polygon in zip(class_ids, normalized_polygons):
                # Ensure the polygon points are flat list of coordinates (x1 y1 x2 y2 ...)
                coords = ' '.join(f'{x:.6f} {y:.6f}' for x, y in polygon)
                yolo_strings.append(f'{class_id} {coords}')
        except Exception as e:
            # Handle cases where results object structure is unexpected
            print(f"Error processing ultralytics results for YOLO conversion: {e}")
            return []

    return yolo_strings


def load_class_names(class_file_path: str) -> List[str]:
    """
    Load class names from a classes.txt file.

    Args:
        class_file_path: Path to classes.txt file

    Returns:
        List of class names, indexed by class ID
    """
    if not os.path.exists(class_file_path):
        return []

    with open(class_file_path, 'r') as f:
        return [line.strip() for line in f if line.strip()]


def save_class_names(class_file_path: str, class_names: List[str]) -> None:
    """
    Save class names to a classes.txt file.

    Args:
        class_file_path: Path to classes.txt file
        class_names: List of class names
    """
    with open(class_file_path, 'w') as f:
        for name in class_names:
            f.write(name + '\n')
