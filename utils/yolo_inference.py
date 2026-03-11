"""
YOLO model inference utilities.
Includes custom ONNX Runtime implementation for specific segmentation models.
"""
import time
import math
import cv2
import numpy as np
from typing import Optional, List, Tuple
# import onnxruntime # Imported lazily or globally? Best globally if installed.
try:
    import onnxruntime
except ImportError:
    onnxruntime = None

# --- Helper Functions from utils.py ---

def nms(boxes:np.ndarray, scores:np.ndarray, iou_threshold:float) -> List[object]:
    # Sort by score
    sorted_indices = np.argsort(scores)[::-1]

    keep_boxes = []
    while sorted_indices.size > 0:
        # Pick the last box
        box_id = sorted_indices[0]
        keep_boxes.append(box_id)

        # Compute IoU of the picked box with the rest
        ious = compute_iou(boxes[box_id, :], boxes[sorted_indices[1:], :])

        # Remove boxes with IoU over the threshold
        keep_indices = np.where(ious < iou_threshold)[0]

        sorted_indices = sorted_indices[keep_indices + 1]

    return keep_boxes


def compute_iou(box: np.ndarray, boxes: np.ndarray) -> np.ndarray:
    # Compute xmin, ymin, xmax, ymax for both boxes
    xmin = np.maximum(box[0], boxes[:, 0])
    ymin = np.maximum(box[1], boxes[:, 1])
    xmax = np.minimum(box[2], boxes[:, 2])
    ymax = np.minimum(box[3], boxes[:, 3])

    # Compute intersection area
    intersection_area = np.maximum(0, xmax - xmin) * np.maximum(0, ymax - ymin)

    # Compute union area
    box_area = (box[2] - box[0]) * (box[3] - box[1])
    boxes_area = (boxes[:, 2] - boxes[:, 0]) * (boxes[:, 3] - boxes[:, 1])
    union_area = box_area + boxes_area - intersection_area

    # Compute IoU
    iou = intersection_area / union_area

    return iou


def xywh2xyxy(x:np.ndarray) -> np.ndarray:
    # Convert bounding box (x, y, w, h) to bounding box (x1, y1, x2, y2)
    y = np.copy(x)
    y[..., 0] = x[..., 0] - x[..., 2] / 2
    y[..., 1] = x[..., 1] - x[..., 3] / 2
    y[..., 2] = x[..., 0] + x[..., 2] / 2
    y[..., 3] = x[..., 1] + x[..., 3] / 2
    return y


def sigmoid(x:np.ndarray)-> np.ndarray:
    return (1 / (1 + np.exp(-x)))

def get_instance_oriented_rect(mask: np.ndarray) -> Optional[List[object]]:
    mask_int = (mask * 255).astype('uint8')
    contours, hierarchy = cv2.findContours(mask_int, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    # Check if contours is empty
    if len(contours) == 0:
        return None

    cnt = contours[0]
    rect = cv2.minAreaRect(cnt)
    
    box_coordinates = cv2.boxPoints(rect)
    corner_pts = box_coordinates.reshape((-1, 1, 2)).astype(np.int32)

    (x, y), width_height, angle = rect
    center_xy_int = (int(x), int(y))

    if angle < -45:
        angle = -int(90 + angle)
    else:
        angle = -int(angle)

    return [corner_pts, center_xy_int, width_height, angle]


def draw_masks_overlay(image: np.ndarray, mask_maps: np.ndarray, class_ids: np.ndarray, alpha=0.4) -> np.ndarray:
    """
    Draw color masks on the image
    """
    overlay = image.copy()
    
    # Define colors (matching draw_OBB)
    color_0 = (255, 0, 0) # Blue
    color_1 = (0, 255, 0) # Green
    
    # If mask_maps is boolean or 0/1, fine.
    
    for i, mask in enumerate(mask_maps):
        if i >= len(class_ids): break
        
        cls = class_ids[i]
        color = color_0 if cls == 0 else color_1
        
        # mask is (H, W), likely uint8 0/1 or similar
        mask_bool = (mask > 0)
        
        # Apply color blending only on masked area
        if mask_bool.any():
            roi = overlay[mask_bool]
            # cv2.addWeighted style blending manually
            # blended = src1 * alpha + src2 * beta + gamma
            # Here: original * (1-alpha) + color * alpha
            
            blended = roi * (1 - alpha) + np.array(color) * alpha
            overlay[mask_bool] = blended.astype(np.uint8)
            
    return overlay

def draw_OBB(image:np.ndarray, mask_info:list[list[object]], class_id: np.ndarray, scores: np.ndarray = None, text=True) -> np.ndarray:
    '''
    mask_info[i] = [corner_pts, center_xy, width_height, angle]
    '''
    # Initialize img_with_instance with a copy of the input image
    img_with_instance = image.copy()
    
    isClosed = True
    line_color_0 = (255, 0, 0)
    line_color_1 = (0, 255, 0)
    line_thickness = 6 
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 1.9 # Increased ~50% from 1.25
    font_color = (255, 255, 255) # Bold White
    font_thickness = 3 # Bold
    bg_color = (255, 0, 0) # Blue background

    for j, info in enumerate(mask_info):
        # info expected structure: [corner_pts, center_xy, width_height, angle]
        
        current_class = class_id[j]
        current_score = scores[j] if scores is not None and len(scores) > j else None
        
        if current_class == 0:
            line_color = line_color_0
        else:
            line_color = line_color_1

        img_with_instance = cv2.polylines(img_with_instance, [info[0]], isClosed, line_color,
                                         line_thickness)

        if text:
            # Info for text: center_xy, angle, class_id, index
            # Lines to print
            lines_to_print = [
                f"Ang: {info[3]}",
                f"Cls: {current_class}"
            ]
            if current_score is not None:
                lines_to_print.append(f"Conf: {current_score:.2f}")

            x, y0 = info[1] # center_xy
            
            # Calculate metrics for the text block to draw background
            max_width = 0
            base_h = 0
            sample_baseline = 0
            
            for line in lines_to_print:
                (w, h), baseline = cv2.getTextSize(str(line), font, font_scale, font_thickness)
                max_width = max(max_width, w)
                base_h = max(base_h, h)
                sample_baseline = max(sample_baseline, baseline)
            
            line_height = int(base_h * 1.5)
            total_block_height = (len(lines_to_print) * line_height)
            
            # Define padding
            pad = 10
            
            # Draw one large rectangle for the block
            # Top-left y: y0 is baseline of first line. So top of box is y0 - h
            # Actually, let's shifting everything so y0 is top-left of BOX to avoid overlapping center point too much? 
            # Original code used center_xy as start point for text.
            
            # Box coords
            box_x1 = x - pad
            box_y1 = y0 - base_h - pad
            box_x2 = x + max_width + pad
            # Bottom is roughly top + total_height + some baseline correction
            box_y2 = box_y1 + total_block_height + pad + sample_baseline
            
            cv2.rectangle(img_with_instance, (box_x1, box_y1), (box_x2, box_y2), bg_color, -1)

            for i, line in enumerate(lines_to_print):
                y = y0 + i * line_height
                cv2.putText(img_with_instance, str(line), (x, y), font, font_scale, font_color,
                            font_thickness)

    return img_with_instance


# --- YOLOSeg Class from YOLOSeg.py ---

class YOLOSeg:

    def __init__(self, path:str, conf_thres=0.5, iou_thres=0.5, num_masks=32, input_width=640, input_height=640)-> None:
        self.conf_threshold = conf_thres
        self.iou_threshold = iou_thres
        self.num_masks = num_masks
        self.input_width = input_width
        self.input_height = input_height
        self.box_predictions = None 
        self.boxes = None
        self.img_height = 0
        self.img_width = 0
        self.pad_w = 0.0
        self.pad_h = 0.0
        self.scale_ratio = 1.0

        # Initialize model
        self.initialize_model(path)

    def __call__(self, image:np.ndarray) -> tuple:
        return self.segment_objects(image)

    def initialize_model(self, path:str) -> None:
        if onnxruntime is None:
            raise ImportError("onnxruntime is not installed.")
            
        self.session = onnxruntime.InferenceSession(path,
                                                    providers=['CUDAExecutionProvider', 'CPUExecutionProvider'])
        # Get model info
        self.get_input_details()
        self.get_output_details()

    def segment_objects(self, image:np.ndarray) -> tuple:
        input_tensor = self.prepare_input(image)

        # Perform inference on the image
        outputs = self.inference(input_tensor)

        self.boxes, self.scores, self.class_ids, mask_pred = self.process_box_output(outputs[0])
        self.mask_maps = self.process_mask_output(mask_pred, outputs[1])

        return self.boxes, self.scores, self.class_ids, self.mask_maps

    def prepare_input(self, image:np.ndarray)->np.ndarray:
        self.img_height, self.img_width = image.shape[:2]

        input_img = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        # Letterboxing to maintain aspect ratio
        self.scale_ratio = min(self.input_width / self.img_width, self.input_height / self.img_height)
        new_unpad_w = int(round(self.img_width * self.scale_ratio))
        new_unpad_h = int(round(self.img_height * self.scale_ratio))
        
        self.pad_w = (self.input_width - new_unpad_w) / 2.0
        self.pad_h = (self.input_height - new_unpad_h) / 2.0

        if (self.img_width, self.img_height) != (new_unpad_w, new_unpad_h):
            input_img = cv2.resize(input_img, (new_unpad_w, new_unpad_h), interpolation=cv2.INTER_LINEAR)
            
        top = int(round(self.pad_h - 0.1))
        bottom = int(round(self.pad_h + 0.1))
        left = int(round(self.pad_w - 0.1))
        right = int(round(self.pad_w + 0.1))
        
        input_img = cv2.copyMakeBorder(input_img, top, bottom, left, right, cv2.BORDER_CONSTANT, value=(114, 114, 114))

        # Scale input pixel values to 0 to 1
        input_img = input_img / 255.0
        input_img = input_img.transpose(2, 0, 1)
        input_tensor = input_img[np.newaxis, :, :, :].astype(np.float32)

        return input_tensor

    def inference(self, input_tensor:np.ndarray) -> List[object]:
        outputs = self.session.run(self.output_names, {self.input_names[0]: input_tensor})
        return outputs

    def process_box_output(self, box_output:np.ndarray) -> tuple:

        predictions = np.squeeze(box_output).T
        num_classes = box_output.shape[1] - self.num_masks - 4

        # Filter out object confidence scores below threshold
        scores = np.max(predictions[:, 4:4+num_classes], axis=1)
        predictions = predictions[scores > self.conf_threshold, :]
        scores = scores[scores > self.conf_threshold]

        if len(scores) == 0:
            return [], [], [], np.array([])

        box_predictions = predictions[..., :num_classes+4]
        mask_predictions = predictions[..., num_classes+4:]

        # Get the class with the highest confidence
        class_ids = np.argmax(box_predictions[:, 4:], axis=1)

        # Get bounding boxes for each object
        boxes = self.extract_boxes(box_predictions)

        # Apply non-maxima suppression to suppress weak, overlapping bounding boxes
        indices = nms(boxes, scores, self.iou_threshold)

        return boxes[indices], scores[indices], class_ids[indices], mask_predictions[indices]

    def process_mask_output(self, mask_predictions:np.ndarray, mask_output:np.ndarray) -> np.ndarray:

        if mask_predictions.shape[0] == 0:
            return []

        mask_output = np.squeeze(mask_output)

        # Calculate the mask maps for each box
        num_mask, mask_height, mask_width = mask_output.shape  # CHW
        masks = sigmoid(mask_predictions @ mask_output.reshape((num_mask, -1)))
        masks = masks.reshape((-1, mask_height, mask_width))

        # Mapping ratio from 640x640 padded input block down to 160x160 mask chunk
        fw = mask_width / self.input_width
        fh = mask_height / self.input_height

        # For every box/mask pair, get the mask map
        mask_maps = np.zeros((len(self.boxes), self.img_height, self.img_width), dtype=np.uint8)
        blur_size = (max(1, int(self.img_width / mask_width)), max(1, int(self.img_height / mask_height)))
        for i in range(len(self.boxes)):

            # Original unpadded image box boundaries
            orig_x1 = self.boxes[i][0]
            orig_y1 = self.boxes[i][1]
            orig_x2 = self.boxes[i][2]
            orig_y2 = self.boxes[i][3]

            # Convert unpadded orig box coords -> 640x640 padded letterbox space coords
            pad_x1 = orig_x1 * self.scale_ratio + self.pad_w
            pad_y1 = orig_y1 * self.scale_ratio + self.pad_h
            pad_x2 = orig_x2 * self.scale_ratio + self.pad_w
            pad_y2 = orig_y2 * self.scale_ratio + self.pad_h

            # Map from 640x640 letterbox space -> 160x160 mask space
            scale_x1 = int(max(0, math.floor(pad_x1 * fw)))
            scale_y1 = int(max(0, math.floor(pad_y1 * fh)))
            scale_x2 = int(min(mask_width, math.ceil(pad_x2 * fw)))
            scale_y2 = int(min(mask_height, math.ceil(pad_y2 * fh)))

            # Original image pixel targets (ints for slice bounds)
            x1 = int(math.floor(orig_x1))
            y1 = int(math.floor(orig_y1))
            x2 = int(math.ceil(orig_x2))
            y2 = int(math.ceil(orig_y2))

            box_w = x2 - x1
            box_h = y2 - y1

            if box_w <= 0 or box_h <= 0 or scale_x2 <= scale_x1 or scale_y2 <= scale_y1:
                continue

            scale_crop_mask = masks[i][scale_y1:scale_y2, scale_x1:scale_x2]
            crop_mask = cv2.resize(scale_crop_mask,
                              (box_w, box_h),
                              interpolation=cv2.INTER_CUBIC)

            crop_mask = cv2.blur(crop_mask, blur_size)

            crop_mask = (crop_mask > 0.5).astype(np.uint8)
            mask_maps[i, y1:y2, x1:x2] = crop_mask

        return mask_maps

    def extract_boxes(self, box_predictions:np.ndarray)->np.ndarray:
        # Extract boxes from predictions (format is [cx, cy, w, h] in 640x640 letterbox coordinates)
        boxes = box_predictions[:, :4]

        # Convert boxes to xyxy format (still in 640x640 space)
        boxes = xywh2xyxy(boxes)

        # Un-pad and rescale back to original unpadded image coordinates
        boxes[:, 0] -= self.pad_w
        boxes[:, 1] -= self.pad_h
        boxes[:, 2] -= self.pad_w
        boxes[:, 3] -= self.pad_h
        boxes[:, :4] /= self.scale_ratio

        # Check the boxes are within the bounds of the original image
        boxes[:, 0] = np.clip(boxes[:, 0], 0, self.img_width)
        boxes[:, 1] = np.clip(boxes[:, 1], 0, self.img_height)
        boxes[:, 2] = np.clip(boxes[:, 2], 0, self.img_width)
        boxes[:, 3] = np.clip(boxes[:, 3], 0, self.img_height)

        return boxes

    def get_input_details(self) -> None:
        model_inputs = self.session.get_inputs()
        self.input_names = [model_inputs[i].name for i in range(len(model_inputs))]
        self.input_shape = model_inputs[0].shape

    def get_output_details(self) -> None:
        model_outputs = self.session.get_outputs()
        self.output_names = [model_outputs[i].name for i in range(len(model_outputs))]

    @staticmethod
    def rescale_boxes(boxes:np.ndarray, input_shape:tuple, image_shape:tuple) -> np.ndarray:
        # Rescale boxes to original image dimensions
        input_shape = np.array([input_shape[1], input_shape[0], input_shape[1], input_shape[0]])
        boxes = np.divide(boxes, input_shape, dtype=np.float32)
        boxes *= np.array([image_shape[1], image_shape[0], image_shape[1], image_shape[0]])

        return boxes


# --- YOLOInference Class (Main Interface) ---

class YOLOInference:
    """Handles YOLO model loading and inference (Supports PT and ONNX via YOLOSeg)"""

    def __init__(self):
        self.models = {}  # Dictionary to store loaded models {slot: model}
        self.item_paths = {} # Dictionary to store model paths {slot: path}
        self.active_slot = 0 # Currently active slot
        self.confidence = 0.5
        self.enabled = True

    def load_model(self, path: str, slot_index: int = 0) -> bool:
        """
        Load a YOLO model from .pt or .onnx file into a specific slot
        """
        try:
            if str(path).lower().endswith('.onnx'):
                # Load using custom YOLOSeg implementation
                print(f"Loading ONNX model: {path}")
                model = YOLOSeg(path, conf_thres=self.confidence)
                self.models[slot_index] = model
            else:
                # Load using Ultralytics
                print(f"Loading PT model: {path}")
                from ultralytics import YOLO
                model = YOLO(path)
                self.models[slot_index] = model

            self.item_paths[slot_index] = path
            return True
        except Exception as e:
            print(f"Error loading model into slot {slot_index}: {e}")
            import traceback
            traceback.print_exc()
            if slot_index in self.models:
                del self.models[slot_index]
            if slot_index in self.item_paths:
                del self.item_paths[slot_index]
            return False

    def unload_model(self, slot_index: int = 0):
        if slot_index in self.models:
            del self.models[slot_index]
        if slot_index in self.item_paths:
            del self.item_paths[slot_index]

    def set_active_slot(self, slot_index: int):
        self.active_slot = slot_index

    def predict(self, frame: np.ndarray):
        """
        Run inference on a frame using the active model
        """
        if not self.enabled:
            return None
            
        model = self.models.get(self.active_slot)
        if model is None:
            return None

        try:
            # Check model type
            if isinstance(model, YOLOSeg):
                # ONNX Inference
                # Update confidence if it changed
                model.conf_threshold = self.confidence
                
                # Run YOLOSeg inference
                # Returns (boxes, scores, class_ids, mask_maps)
                return model(frame)
            else:
                # PT Inference (Ultralytics)
                results = model(frame, conf=self.confidence, verbose=False)
                return results[0] if results else None
                
        except Exception as e:
            print(f"Error during inference: {e}")
            import traceback
            traceback.print_exc()
            return None

    def draw_results(self, frame: np.ndarray, results) -> np.ndarray:
        """
        Draw results on frame. Handles both Ultralytics Results and ONNX tuples.
        """
        if results is None:
            return frame

        try:
            if isinstance(results, tuple):
                # ONNX Results: (boxes, scores, class_ids, mask_maps)
                boxes, scores, class_ids, mask_maps = results
                
                # 0. Draw masks first (underlay)
                if mask_maps is not None:
                     frame = draw_masks_overlay(frame, mask_maps, class_ids)
                
                # 1. Draw OBB (Oriented Bounding Boxes)
                instance_rect_list = []
                valid_class_ids = []
                valid_scores = []
                
                if len(scores) > 0:
                    for i in range(len(scores)):
                        # Get oriented rect for each mask
                        if mask_maps is not None and len(mask_maps) > i:
                            instance_rect = get_instance_oriented_rect(mask_maps[i])
                            if instance_rect is not None:
                                instance_rect_list.append(instance_rect)
                                valid_class_ids.append(class_ids[i])
                                valid_scores.append(scores[i])
                
                if instance_rect_list:
                    # Draw OBBs with scores
                    annotated_frame = draw_OBB(frame, instance_rect_list, np.array(valid_class_ids), np.array(valid_scores))
                    return annotated_frame
                else:
                    return frame

            else:
                # Ultralytics Results
                return results.plot()
                
        except Exception as e:
            print(f"Error drawing results: {e}")
            import traceback
            traceback.print_exc()
            return frame

    def set_confidence(self, value: float):
        self.confidence = max(0.0, min(1.0, value))

    def set_enabled(self, enabled: bool):
        self.enabled = enabled

    def is_loaded(self, slot_index: Optional[int] = None) -> bool:
        target_slot = slot_index if slot_index is not None else self.active_slot
        return target_slot in self.models

    def get_model_path(self, slot_index: Optional[int] = None) -> Optional[str]:
        target_slot = slot_index if slot_index is not None else self.active_slot
        return self.item_paths.get(target_slot)
