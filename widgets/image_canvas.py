"""
Image canvas widget for displaying and annotating images.
"""
from typing import List, Optional, Tuple
from PySide6.QtWidgets import QGraphicsView, QGraphicsScene, QGraphicsPixmapItem, QGraphicsPolygonItem, QGraphicsEllipseItem, QGraphicsPathItem
from PySide6.QtCore import Qt, Signal, QPointF, QRectF
from PySide6.QtGui import QPixmap, QImage, QPen, QBrush, QColor, QPolygonF, QPainter, QPainterPath
from PIL import Image
import cv2
import numpy as np
from models.annotation import Annotation


class ImageCanvas(QGraphicsView):
    """Canvas for displaying images and handling annotation interactions"""

    # Signals
    annotation_added = Signal(Annotation)
    annotation_modified = Signal()
    annotation_deleted = Signal(Annotation)
    annotation_selected = Signal(Annotation)  # Emitted when an annotation is selected

    def __init__(self, parent=None):
        super().__init__(parent)

        # Setup scene
        self.scene = QGraphicsScene(self)
        self.setScene(self.scene)

        # Setup view properties
        self.setRenderHint(QPainter.Antialiasing)
        self.setRenderHint(QPainter.SmoothPixmapTransform)
        self.setDragMode(QGraphicsView.NoDrag)
        self.setHorizontalScrollBarPolicy(Qt.ScrollBarAsNeeded)
        self.setVerticalScrollBarPolicy(Qt.ScrollBarAsNeeded)

        # Image data
        self.pixmap_item: Optional[QGraphicsPixmapItem] = None
        self.image_path: str = ""
        self.image_width: int = 0
        self.image_height: int = 0
        
        # Image Enhancement State
        self.original_image_np: Optional[np.ndarray] = None
        self.brightness = 1.0
        self.contrast = 1.0
        self.gamma = 1.0
        self.use_clahe = False
        self.invert_colors = False

        # Annotations
        self.annotations: List[Annotation] = []
        self.show_annotations: bool = True

        # Drawing state
        self.annotation_mode: str = "segmentation"
        self.drawing_mode: bool = False
        self.current_polygon: List[Tuple[float, float]] = []  # Pixel coordinates
        self.current_class_id: int = 0
        self.current_class_name: str = ""
        self.shift_pressed: bool = False  # Track Shift key state during polygon drawing

        # Editing state
        self.editing_mode: bool = False
        self.selected_annotation: Optional[Annotation] = None
        self.dragging_vertex: bool = False
        self.dragging_vertex_index: int = -1

        # Graphics items for visualization
        self.polygon_items: List[QGraphicsPolygonItem] = []
        self.vertex_items: List[List[QGraphicsPathItem]] = []
        self.temp_polygon_item: Optional[QGraphicsPolygonItem] = None
        self.temp_vertex_items: List[QGraphicsPathItem] = []

    def load_image(self, image_path: str) -> bool:
        """Load an image from file path"""
        try:
            # Load image using PIL to handle various formats
            pil_image = Image.open(image_path)
            pil_image = pil_image.convert('RGB')
            self.original_image_np = np.array(pil_image)

            self.image_path = image_path
            self.image_width = pil_image.width
            self.image_height = pil_image.height

            # Clear scene
            self.scene.clear()
            self.polygon_items.clear()
            self.vertex_items.clear()

            # Apply enhancements and display
            self.apply_enhancements(initial_load=True)

            return True

        except Exception as e:
            print(f"Error loading image {image_path}: {e}")
            return False

    def set_enhancements(self, brightness=1.0, contrast=1.0, gamma=1.0, use_clahe=False, invert_colors=False):
        """Update enhancement parameters and re-apply."""
        self.brightness = brightness
        self.contrast = contrast
        self.gamma = gamma
        self.use_clahe = use_clahe
        self.invert_colors = invert_colors
        self.apply_enhancements()

    def apply_enhancements(self, initial_load=False):
        """Apply enhancements to the original image and update the canvas."""
        if self.original_image_np is None:
            return

        img = self.original_image_np.copy()

        # 1. CLAHE (Contrast Limited Adaptive Histogram Equalization)
        if self.use_clahe:
            lab = cv2.cvtColor(img, cv2.COLOR_RGB2LAB)
            l_channel, a, b = cv2.split(lab)
            clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
            cl = clahe.apply(l_channel)
            limg = cv2.merge((cl, a, b))
            img = cv2.cvtColor(limg, cv2.COLOR_LAB2RGB)

        # 2. Gamma Correction
        if self.gamma != 1.0:
            inv_gamma = 1.0 / self.gamma
            table = np.array([((i / 255.0) ** inv_gamma) * 255
                              for i in np.arange(0, 256)]).astype("uint8")
            img = cv2.LUT(img, table)

        # 3. Brightness and Contrast
        # contrast -> alpha [0.1, 3.0]
        # brightness -> beta [-127.5, 127.5] where 1.0 is 0
        beta = (self.brightness - 1.0) * 127.5
        if self.contrast != 1.0 or beta != 0:
            img = cv2.convertScaleAbs(img, alpha=self.contrast, beta=beta)

        # 4. Invert Colors
        if self.invert_colors:
            img = cv2.bitwise_not(img)

        # Convert to QImage
        height, width, channel = img.shape
        bytes_per_line = 3 * width
        
        # Ensure array is contiguous and has the standard RGB layout required by QImage
        # Creating a copy ensures the memory is properly aligned for PySide6
        img = np.ascontiguousarray(img)
        
        qimage = QImage(img.data, width, height, bytes_per_line, QImage.Format_RGB888)
        pixmap = QPixmap.fromImage(qimage)

        if initial_load:
            self.pixmap_item = self.scene.addPixmap(pixmap)
            self.fitInView(self.pixmap_item, Qt.KeepAspectRatio)
        else:
            if self.pixmap_item:
                self.pixmap_item.setPixmap(pixmap)

    def set_annotations(self, annotations: List[Annotation]) -> None:
        """Set the annotations to display"""
        self.annotations = annotations
        self.redraw_annotations()

    def redraw_annotations(self) -> None:
        """Redraw all annotations on the canvas"""
        # Clear existing annotation graphics
        for item in self.polygon_items:
            self.scene.removeItem(item)
        for vertex_list in self.vertex_items:
            for item in vertex_list:
                self.scene.removeItem(item)

        self.polygon_items.clear()
        self.vertex_items.clear()

        if not self.show_annotations or not self.pixmap_item:
            return

        # Draw each annotation
        for annotation in self.annotations:
            self._draw_annotation(annotation)

    def _draw_annotation(self, annotation: Annotation) -> None:
        """Draw a single annotation"""
        if not self.pixmap_item:
            return

        # Get pixel points
        pixel_points = annotation.get_pixel_points(self.image_width, self.image_height)

        # Create polygon
        polygon = QPolygonF(pixel_points)

        # Set pen and brush based on selection state
        if annotation.selected:
            pen = QPen(QColor(255, 255, 0), 3)  # Yellow for selected
            if self.annotation_mode == "detection":
                brush = QBrush(Qt.transparent)
            else:
                brush = QBrush(QColor(255, 255, 0, 80))
        else:
            pen = QPen(annotation.color.darker(150), 2)
            if self.annotation_mode == "detection":
                brush = QBrush(Qt.transparent)
            else:
                brush = QBrush(annotation.color)

        # Add polygon to scene
        polygon_item = self.scene.addPolygon(polygon, pen, brush)
        polygon_item.setZValue(1)  # Above image
        self.polygon_items.append(polygon_item)

        # Draw crosshair vertices
        vertex_list = []
        if annotation.selected:
            pen = QPen(Qt.yellow, 1)
        else:
            pen = QPen(QColor(annotation.color.red(), annotation.color.green(), annotation.color.blue()), 1)

        for point in pixel_points:
            if self.annotation_mode == "detection":
                vertex_item = self._create_solid_dot_item(point.x(), point.y(), 1.2, pen.color())
            else:
                vertex_item = self._create_crosshair_item(point.x(), point.y(), 12, pen)
            vertex_list.append(vertex_item)

        self.vertex_items.append(vertex_list)

    def toggle_annotation_visibility(self) -> None:
        """Toggle the visibility of annotations"""
        self.show_annotations = not self.show_annotations
        self.redraw_annotations()

    def start_drawing(self, class_id: int, class_name: str = "") -> None:
        """Start drawing a new polygon annotation"""
        self.drawing_mode = True
        self.current_polygon = []
        self.current_class_id = class_id
        self.current_class_name = class_name
        self.setCursor(Qt.CrossCursor)

    def stop_drawing(self) -> None:
        """Stop drawing mode and cancel current polygon"""
        self.drawing_mode = False
        self.current_polygon = []
        self.shift_pressed = False
        self._clear_temp_graphics()
        self.setCursor(Qt.ArrowCursor)

    def set_annotation_mode(self, mode: str) -> None:
        """Set the annotation mode (segmentation or detection)"""
        self.annotation_mode = mode
        if self.drawing_mode:
            self.stop_drawing()

    def finish_polygon(self) -> None:
        """Finish the current polygon and create an annotation"""
        if self.annotation_mode == "segmentation":
            if len(self.current_polygon) < 3:
                print("Polygon needs at least 3 points")
                self.stop_drawing()
                return
            points = self.current_polygon
        else:
            if len(self.current_polygon) != 2:
                self.stop_drawing()
                return
            p1, p2 = self.current_polygon
            if abs(p1[0] - p2[0]) < 5 or abs(p1[1] - p2[1]) < 5:
                print("Bounding box too small")
                self.stop_drawing()
                return
            # Convert 2 points to 4 points polygon
            points = [
                (p1[0], p1[1]),
                (p2[0], p1[1]),
                (p2[0], p2[1]),
                (p1[0], p2[1])
            ]

        # Create annotation from pixel coordinates
        normalized_points = [
            (x / self.image_width, y / self.image_height)
            for x, y in points
        ]

        annotation = Annotation(self.current_class_id, normalized_points, self.current_class_name)
        self.annotations.append(annotation)
        self.annotation_added.emit(annotation)

        # Clear drawing state
        self.stop_drawing()
        self.redraw_annotations()

    def _clear_temp_graphics(self) -> None:
        """Clear temporary graphics (current drawing)"""
        if self.temp_polygon_item:
            self.scene.removeItem(self.temp_polygon_item)
            self.temp_polygon_item = None

        for item in self.temp_vertex_items:
            self.scene.removeItem(item)
        self.temp_vertex_items.clear()

    def _draw_temp_polygon(self) -> None:
        """Draw the polygon currently being created"""
        self._clear_temp_graphics()

        if self.annotation_mode == "segmentation":
            # Draw polygon lines/fill only if we have at least 2 points
            if len(self.current_polygon) >= 2:
                points = [QPointF(x, y) for x, y in self.current_polygon]
                polygon = QPolygonF(points)
                self.temp_polygon_item = self.scene.addPolygon(
                    polygon,
                    QPen(QColor(0, 255, 0), 2),
                    QBrush(QColor(0, 255, 0, 80))
                )
                self.temp_polygon_item.setZValue(1)

            # Draw crosshair vertices for ALL points (including the first one)
            pen = QPen(Qt.green, 1)
            for x, y in self.current_polygon:
                vertex_item = self._create_crosshair_item(x, y, 12, pen)
                self.temp_vertex_items.append(vertex_item)
        else:
            if len(self.current_polygon) == 2:
                p1, p2 = self.current_polygon
                points = [
                    QPointF(p1[0], p1[1]),
                    QPointF(p2[0], p1[1]),
                    QPointF(p2[0], p2[1]),
                    QPointF(p1[0], p2[1])
                ]
                polygon = QPolygonF(points)
                self.temp_polygon_item = self.scene.addPolygon(
                    polygon,
                    QPen(QColor(0, 255, 0), 1), # Reduced from 2 to 1 (50% thinner)
                    QBrush(Qt.transparent)
                )
                self.temp_polygon_item.setZValue(1)
                
                # Draw solid dot vertices for the 4 corners
                for p in points:
                    # 10% of radius 12 is 1.2
                    vertex_item = self._create_solid_dot_item(p.x(), p.y(), 1.2, QColor(0, 255, 0))
                    self.temp_vertex_items.append(vertex_item)

    def _create_solid_dot_item(self, x: float, y: float, radius: float, color: QColor) -> QGraphicsEllipseItem:
        """Create a solid dot graphics item"""
        item = self.scene.addEllipse(x - radius, y - radius, radius * 2, radius * 2, QPen(Qt.transparent), QBrush(color))
        item.setZValue(2)
        return item

    def _create_crosshair_item(self, x: float, y: float, radius: int, pen: QPen) -> QGraphicsPathItem:
        """Create a crosshair graphics item (transparent circle + cross lines)"""
        path = QPainterPath()
        # Circle
        path.addEllipse(QPointF(x, y), radius, radius)
        # Horizontal line
        path.moveTo(x - radius, y)
        path.lineTo(x + radius, y)
        # Vertical line
        path.moveTo(x, y - radius)
        path.lineTo(x, y + radius)

        item = self.scene.addPath(path, pen, QBrush(Qt.transparent))
        item.setZValue(2)  # Above polygon
        return item

    def set_current_class(self, class_id: int, class_name: str) -> None:
        """Set the current class for new annotations"""
        self.current_class_id = class_id
        self.current_class_name = class_name

    def delete_selected_annotation(self) -> None:
        """Delete the currently selected annotation"""
        if self.selected_annotation and self.selected_annotation in self.annotations:
            self.annotations.remove(self.selected_annotation)
            self.annotation_deleted.emit(self.selected_annotation)
            self.selected_annotation = None
            self.redraw_annotations()

    def select_annotation_by_index(self, index: int) -> None:
        """Select an annotation by its index in the annotations list"""
        if 0 <= index < len(self.annotations):
            # Deselect all
            for ann in self.annotations:
                ann.selected = False
            # Select the one at the given index
            self.annotations[index].selected = True
            self.selected_annotation = self.annotations[index]
            self.redraw_annotations()
            self.annotation_selected.emit(self.selected_annotation)

    def mousePressEvent(self, event):
        """Handle mouse press events"""
        if event.button() == Qt.MiddleButton:
            self.setDragMode(QGraphicsView.ScrollHandDrag)
            # Create a fake left click event so ScrollHandDrag works properly
            from PySide6.QtGui import QMouseEvent
            fake_event = QMouseEvent(event.type(), event.pos(), event.globalPos(),
                                     Qt.LeftButton, Qt.LeftButton, event.modifiers())
            super().mousePressEvent(fake_event)
            return

        if not self.pixmap_item:
            return

        # Map to scene coordinates
        scene_pos = self.mapToScene(event.pos())
        x, y = scene_pos.x(), scene_pos.y()

        # Check if click is within image bounds
        if not (0 <= x <= self.image_width and 0 <= y <= self.image_height):
            super().mousePressEvent(event)
            return

        if event.button() == Qt.LeftButton:
            # Check if clicking on a vertex of selected annotation
            if self.selected_annotation:
                vertex_index = self.selected_annotation.get_nearest_vertex(
                    (x, y), self.image_width, self.image_height
                )
                if self.selected_annotation.contains_point((x, y), self.image_width, self.image_height, tolerance=10):
                    self.dragging_vertex = True
                    self.dragging_vertex_index = vertex_index
                    return

            # Try to select an annotation
            selected = False
            for annotation in reversed(self.annotations):  # Check from top to bottom
                if annotation.is_inside_polygon((x, y), self.image_width, self.image_height):
                    # Deselect all
                    for ann in self.annotations:
                        ann.selected = False
                    # Select this one
                    annotation.selected = True
                    self.selected_annotation = annotation
                    selected = True
                    self.redraw_annotations()
                    self.annotation_selected.emit(annotation)  # Emit selection signal
                    break

            if not selected:
                # Deselect all
                for ann in self.annotations:
                    ann.selected = False
                self.selected_annotation = None
                self.redraw_annotations()

            if self.annotation_mode == "segmentation":
                # Check for Shift+Click to add points (start drawing if needed)
                if event.modifiers() & Qt.ShiftModifier:
                    if not self.drawing_mode:
                        self.start_drawing(self.current_class_id, self.current_class_name)
                    
                    self.current_polygon.append((x, y))
                    self._draw_temp_polygon()
                    self.shift_pressed = True
                    return

                if self.drawing_mode:
                    return
            else:
                if event.modifiers() & Qt.ShiftModifier:
                    if not self.drawing_mode and not selected and not self.dragging_vertex:
                        self.start_drawing(self.current_class_id, self.current_class_name)
                        self.current_polygon = [(x, y), (x, y)]
                        self._draw_temp_polygon()
                        self.shift_pressed = True
                        return

                if self.drawing_mode:
                    return

        super().mousePressEvent(event)

    def mouseMoveEvent(self, event):
        """Handle mouse move events"""
        scene_pos = self.mapToScene(event.pos())
        x, y = scene_pos.x(), scene_pos.y()
        # Clamp to image bounds
        x = max(0, min(self.image_width, x))
        y = max(0, min(self.image_height, y))

        if self.dragging_vertex and self.selected_annotation:
            if self.annotation_mode == "detection" and len(self.selected_annotation.points) == 4:
                idx = self.dragging_vertex_index
                pixel_points = self.selected_annotation.get_pixel_points(self.image_width, self.image_height)
                
                p0 = [pixel_points[0].x(), pixel_points[0].y()]
                p1 = [pixel_points[1].x(), pixel_points[1].y()]
                p2 = [pixel_points[2].x(), pixel_points[2].y()]
                p3 = [pixel_points[3].x(), pixel_points[3].y()]

                if idx == 0:
                    p0 = [x, y]
                    p1[1] = y
                    p3[0] = x
                elif idx == 1:
                    p1 = [x, y]
                    p0[1] = y
                    p2[0] = x
                elif idx == 2:
                    p2 = [x, y]
                    p1[0] = x
                    p3[1] = y
                elif idx == 3:
                    p3 = [x, y]
                    p0[0] = x
                    p2[1] = y

                self.selected_annotation.update_vertex(0, tuple(p0), self.image_width, self.image_height)
                self.selected_annotation.update_vertex(1, tuple(p1), self.image_width, self.image_height)
                self.selected_annotation.update_vertex(2, tuple(p2), self.image_width, self.image_height)
                self.selected_annotation.update_vertex(3, tuple(p3), self.image_width, self.image_height)
            else:
                # Update vertex
                self.selected_annotation.update_vertex(
                    self.dragging_vertex_index,
                    (x, y),
                    self.image_width,
                    self.image_height
                )
            self.redraw_annotations()
            # Don't emit annotation_modified here - wait until mouse release
        elif self.drawing_mode and self.annotation_mode == "detection" and len(self.current_polygon) == 2:
            self.current_polygon[1] = (x, y)
            self._draw_temp_polygon()

        super().mouseMoveEvent(event)

    def mouseReleaseEvent(self, event):
        """Handle mouse release events"""
        if event.button() == Qt.MiddleButton:
            self.setDragMode(QGraphicsView.NoDrag)
            from PySide6.QtGui import QMouseEvent
            fake_event = QMouseEvent(event.type(), event.pos(), event.globalPos(),
                                     Qt.LeftButton, Qt.LeftButton, event.modifiers())
            super().mouseReleaseEvent(fake_event)
            return

        if event.button() == Qt.LeftButton:
            # Emit modification signal when drag is complete
            if self.dragging_vertex and self.selected_annotation:
                self.annotation_modified.emit()

            self.dragging_vertex = False
            self.dragging_vertex_index = -1
            
            if self.drawing_mode and self.annotation_mode == "detection":
                self.finish_polygon()

        super().mouseReleaseEvent(event)

    def mouseDoubleClickEvent(self, event):
        """Handle double-click to finish polygon"""
        if self.drawing_mode and event.button() == Qt.LeftButton:
            self.finish_polygon()
        super().mouseDoubleClickEvent(event)

    def wheelEvent(self, event):
        """Handle mouse wheel events for zooming"""
        if event.modifiers() & Qt.ControlModifier:
            # Calculate zoom factor
            zoom_in_factor = 1.15
            zoom_out_factor = 1.0 / zoom_in_factor

            # Set anchor to mouse position
            self.setTransformationAnchor(QGraphicsView.AnchorUnderMouse)

            # Zoom
            # angleDelta().y() is usually 120 (up/forward) or -120 (down/backward)
            if event.angleDelta().y() > 0:
                self.scale(zoom_in_factor, zoom_in_factor)
            elif event.angleDelta().y() < 0:
                self.scale(zoom_out_factor, zoom_out_factor)
            
            # Prevent event from being processed by scrollbars
            event.accept()
        else:
            super().wheelEvent(event)

    def keyPressEvent(self, event):
        """Handle key press events"""
        # Let arrow keys and space propagate to parent (main window)
        if event.key() in (Qt.Key_Left, Qt.Key_Right, Qt.Key_Space):
            event.ignore()  # Propagate to parent
            return

        if event.key() == Qt.Key_Escape:
            if self.drawing_mode:
                self.stop_drawing()
            elif self.selected_annotation:
                for ann in self.annotations:
                    ann.selected = False
                self.selected_annotation = None
                self.redraw_annotations()

        elif event.key() == Qt.Key_Return or event.key() == Qt.Key_Enter:
            if self.drawing_mode:
                self.finish_polygon()

        elif event.key() == Qt.Key_Delete:
            # If Shift+Delete, ignore it so parent (MainWindow) can handle "Delete All"
            if event.modifiers() & Qt.ShiftModifier:
                event.ignore()
                return
            self.delete_selected_annotation()

        else:
            super().keyPressEvent(event)

    def keyReleaseEvent(self, event):
        """Handle key release events"""
        # When Shift is released during drawing, finish the polygon or bbox
        if event.key() == Qt.Key_Shift and self.drawing_mode and self.shift_pressed:
            self.finish_polygon()
        else:
            super().keyReleaseEvent(event)

    def resizeEvent(self, event):
        """Handle resize events"""
        super().resizeEvent(event)
        # Prevent forcing fitInView on resize so that zoom level isn't ruined
        # if self.pixmap_item:
        #     self.fitInView(self.pixmap_item, Qt.KeepAspectRatio)
