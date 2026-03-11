import os
import cv2
import numpy as np
import shutil
import csv
import matplotlib.pyplot as plt
from PySide6.QtWidgets import (QWidget, QVBoxLayout, QHBoxLayout, QPushButton,
                               QLabel, QFileDialog, QProgressBar, QMessageBox,
                               QGroupBox, QFormLayout, QDoubleSpinBox)
from PySide6.QtCore import Qt, QThread, Signal

from utils.yolo_inference import YOLOInference
from utils.yolo_format import load_annotations


class SearchThread(QThread):
    progress = Signal(int, int)  # current, total
    finished = Signal()
    error = Signal(str)
    log = Signal(str)

    def __init__(self, img_dir, lbl_dir, out_dir, model_path, iou_threshold, min_instances, max_instances):
        super().__init__()
        self.img_dir = img_dir
        self.lbl_dir = lbl_dir
        self.out_dir = out_dir
        self.model_path = model_path
        self.iou_threshold = iou_threshold
        self.min_instances = min_instances
        self.max_instances = max_instances
        self._is_running = True

    def run(self):
        try:
            self.log.emit("Initializing inference engine...")
            inference_engine = YOLOInference()
            if not inference_engine.load_model(self.model_path, 0):
                self.error.emit(f"Failed to load model: {self.model_path}")
                return

            self.log.emit("Scanning directories...")
            img_files = sorted([f for f in os.listdir(self.img_dir) if f.lower().endswith(('.png', '.jpg', '.jpeg'))])
            
            if not img_files:
                self.error.emit("No images found in the source image directory.")
                return

            out_imgs = os.path.join(self.out_dir, "images")
            out_lbls = os.path.join(self.out_dir, "gt labels")
            out_preds = os.path.join(self.out_dir, "predicted")

            os.makedirs(out_imgs, exist_ok=True)
            os.makedirs(out_lbls, exist_ok=True)
            os.makedirs(out_preds, exist_ok=True)

            total = len(img_files)
            
            # List to store results for CSV and charting
            results_data = []

            for i, img_name in enumerate(img_files):
                if not self._is_running:
                    self.log.emit("Search stopped by user.")
                    break

                self.progress.emit(i, total)

                base_name = os.path.splitext(img_name)[0]
                lbl_name = base_name + ".txt"
                lbl_path = os.path.join(self.lbl_dir, lbl_name)
                img_path = os.path.join(self.img_dir, img_name)

                # If label doesn't exist, we assume no objects (empty ground truth)
                annotations = load_annotations(lbl_path) if os.path.exists(lbl_path) else []

                frame = cv2.imread(img_path)
                if frame is None:
                    continue

                h, w = frame.shape[:2]

                # GT Mask
                gt_mask = np.zeros((h, w), dtype=np.uint8)
                for ann in annotations:
                    pts = ann.to_pixel_coords(w, h)
                    if pts:
                        pts = np.array(pts, dtype=np.int32).reshape((-1, 1, 2))
                        cv2.fillPoly(gt_mask, [pts], 1)

                # Prediction Mask and Instances Count
                results = inference_engine.predict(frame)
                pred_mask = np.zeros((h, w), dtype=np.uint8)
                num_instances = 0

                if results is not None:
                    if isinstance(results, tuple):  # ONNX
                        boxes, _, _, mask_maps = results
                        if boxes is not None:
                             num_instances = len(boxes)
                        if mask_maps is not None and len(mask_maps) > 0:
                            pred_mask = np.max(mask_maps, axis=0).astype(np.uint8)
                    else:  # PT
                        if hasattr(results, 'boxes') and results.boxes is not None:
                             num_instances = len(results.boxes)
                        if hasattr(results, 'masks') and results.masks is not None:
                            for polygon in results.masks.xy:
                                pts = np.array(polygon, dtype=np.int32).reshape((-1, 1, 2))
                                cv2.fillPoly(pred_mask, [pts], 1)

                # Calculate IoU
                intersection = np.logical_and(gt_mask, pred_mask).sum()
                union = np.logical_or(gt_mask, pred_mask).sum()

                if union == 0:
                    iou = 1.0  # Both empty -> perfect match
                else:
                    iou = intersection / union

                # Store result
                results_data.append((img_name, iou, num_instances))

                # If IoU < threshold OR instances out of bounds, save to output
                is_failed_iou = iou < self.iou_threshold
                is_out_of_bounds = not (self.min_instances <= num_instances <= self.max_instances)

                if is_failed_iou or is_out_of_bounds:
                    reason = f"IoU: {iou:.3f}" if is_failed_iou else f"Instances: {num_instances} (Allowed: {self.min_instances}-{self.max_instances})"
                    self.log.emit(f"Found {img_name} - {reason}")
                    
                    # Copy original image and label
                    shutil.copy2(img_path, os.path.join(out_imgs, img_name))
                    if os.path.exists(lbl_path):
                        shutil.copy2(lbl_path, os.path.join(out_lbls, lbl_name))
                    else:
                        open(os.path.join(out_lbls, lbl_name), 'w').close()

                    # Generate comparison image
                    gt_frame = frame.copy()
                    overlay = gt_frame.copy()
                    
                    # Draw GT on left
                    for ann in annotations:
                        pts = ann.to_pixel_coords(w, h)
                        if pts:
                            pts = np.array(pts, dtype=np.int32).reshape((-1, 1, 2))
                            cv2.fillPoly(overlay, [pts], (0, 255, 0))
                    cv2.addWeighted(overlay, 0.4, gt_frame, 0.6, 0, gt_frame)
                    cv2.putText(gt_frame, "Ground Truth", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
                    
                    # Draw Pred on right
                    pred_frame = inference_engine.draw_results(frame.copy(), results)
                    cv2.putText(pred_frame, f"Pred (IoU:{iou:.2f}, Inst:{num_instances})", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)

                    # Create a gap between images
                    gap_width = 20
                    gap = np.zeros((h, gap_width, 3), dtype=np.uint8)
                    gap[:] = (147, 20, 255) # Pink background (BGR)

                    # Draw dashed line (white dots) in the middle of the gap
                    dash_length = 10
                    dash_gap = 10
                    x_center = gap_width // 2
                    for y in range(0, h, dash_length + dash_gap):
                        cv2.line(gap, (x_center, y), (x_center, min(y + dash_length, h)), (255, 255, 255), 2)

                    compare_img = np.hstack((gt_frame, gap, pred_frame))
                    cv2.imwrite(os.path.join(out_preds, img_name), compare_img)

            # --- Post-loop Analysis (CSV and Chart) ---
            if results_data:
                self.log.emit("Generating CSV and charts...")
                
                # Sort by IoU ascending
                results_data.sort(key=lambda x: x[1])

                # Save CSV
                csv_path = os.path.join(self.out_dir, "iou_results.csv")
                with open(csv_path, 'w', newline='') as f:
                    writer = csv.writer(f)
                    writer.writerow(["Image Name", "IoU", "Num Instances"])
                    writer.writerows(results_data)

                # Generate Graph
                # x is the iou threshold from 0 to 1, y is the cumulative number of images below that threshold
                thresholds = np.linspace(0.0, 1.0, 101) # 0.00 to 1.00
                ious = np.array([res[1] for res in results_data])
                
                cumulative_counts = [np.sum(ious < t) for t in thresholds]

                plt.figure(figsize=(10, 6))
                plt.plot(thresholds, cumulative_counts, marker='', color='b', linewidth=2)
                plt.title('Cumulative Number of Images Below IoU Threshold')
                plt.xlabel('IoU Threshold')
                plt.ylabel('Number of Images')
                plt.grid(True, linestyle='--', alpha=0.7)
                plt.tight_layout()
                
                graph_path = os.path.join(self.out_dir, "iou_cumulative_graph.png")
                plt.savefig(graph_path)
                plt.close()

            self.progress.emit(total, total)
            self.log.emit("Search completed.")
            self.finished.emit()

        except Exception as e:
            import traceback
            traceback.print_exc()
            self.error.emit(str(e))

    def stop(self):
        self._is_running = False


class FindRetrainingDatasetTab(QWidget):
    def __init__(self):
        super().__init__()
        self.search_thread = None
        self._setup_ui()

    def _setup_ui(self):
        main_layout = QVBoxLayout(self)

        group_box = QGroupBox("Find Retraining Dataset Configurations")
        form_layout = QFormLayout()

        # Source Images Directory
        self.src_imgs_lbl = QLabel("Not selected")
        self.src_imgs_lbl.setStyleSheet("color: gray;")
        src_imgs_btn = QPushButton("Select")
        src_imgs_btn.setFixedWidth(100)
        src_imgs_btn.clicked.connect(self._select_src_images)
        imgs_layout = QHBoxLayout()
        imgs_layout.addWidget(src_imgs_btn)
        imgs_layout.addWidget(self.src_imgs_lbl)
        imgs_layout.addStretch()
        form_layout.addRow("Src Image Dir:", imgs_layout)

        # Source Labels Directory
        self.src_lbls_lbl = QLabel("Not selected")
        self.src_lbls_lbl.setStyleSheet("color: gray;")
        src_lbls_btn = QPushButton("Select")
        src_lbls_btn.setFixedWidth(100)
        src_lbls_btn.clicked.connect(self._select_src_labels)
        lbls_layout = QHBoxLayout()
        lbls_layout.addWidget(src_lbls_btn)
        lbls_layout.addWidget(self.src_lbls_lbl)
        lbls_layout.addStretch()
        form_layout.addRow("Src Groundtruth label dir:", lbls_layout)

        # Output Directory
        self.out_dir_lbl = QLabel("Not selected")
        self.out_dir_lbl.setStyleSheet("color: gray;")
        out_dir_btn = QPushButton("Select")
        out_dir_btn.setFixedWidth(100)
        out_dir_btn.clicked.connect(self._select_out_dir)
        out_layout = QHBoxLayout()
        out_layout.addWidget(out_dir_btn)
        out_layout.addWidget(self.out_dir_lbl)
        out_layout.addStretch()
        form_layout.addRow("Output Path:", out_layout)

        # Model Selection
        self.model_lbl = QLabel("Not selected")
        self.model_lbl.setStyleSheet("color: gray;")
        model_btn = QPushButton("Select")
        model_btn.setFixedWidth(100)
        model_btn.clicked.connect(self._select_model)
        model_layout = QHBoxLayout()
        model_layout.addWidget(model_btn)
        model_layout.addWidget(self.model_lbl)
        model_layout.addStretch()
        form_layout.addRow("YOLO Model:", model_layout)

        # IoU Threshold Input
        self.iou_spinbox = QDoubleSpinBox()
        self.iou_spinbox.setRange(0.0, 1.0)
        self.iou_spinbox.setSingleStep(0.05)
        self.iou_spinbox.setValue(0.50)
        self.iou_spinbox.setFixedWidth(100)
        form_layout.addRow("IoU Threshold:", self.iou_spinbox)

        # Min Instances
        from PySide6.QtWidgets import QSpinBox
        self.min_inst_spinbox = QSpinBox()
        self.min_inst_spinbox.setRange(0, 9999)
        self.min_inst_spinbox.setValue(0)
        self.min_inst_spinbox.setFixedWidth(100)
        form_layout.addRow("Min Instances:", self.min_inst_spinbox)

        # Max Instances
        self.max_inst_spinbox = QSpinBox()
        self.max_inst_spinbox.setRange(0, 9999)
        self.max_inst_spinbox.setValue(1)
        self.max_inst_spinbox.setFixedWidth(100)
        form_layout.addRow("Max Instances:", self.max_inst_spinbox)

        group_box.setLayout(form_layout)
        main_layout.addWidget(group_box)

        # Controls & Progress
        self.search_btn = QPushButton("Search dataset")
        self.search_btn.clicked.connect(self._toggle_search)
        main_layout.addWidget(self.search_btn)

        self.progress_bar = QProgressBar()
        self.progress_bar.setValue(0)
        main_layout.addWidget(self.progress_bar)

        self.status_lbl = QLabel("Ready.")
        main_layout.addWidget(self.status_lbl)

        main_layout.addStretch()

        self.src_imgs_dir = ""
        self.src_lbls_dir = ""
        self.out_dir = ""
        self.model_path = ""

    def _select_src_images(self):
        folder = QFileDialog.getExistingDirectory(self, "Select Source Images Directory")
        if folder:
            self.src_imgs_dir = folder
            self.src_imgs_lbl.setText(folder)
            self.src_imgs_lbl.setStyleSheet("color: black;")

    def _select_src_labels(self):
        folder = QFileDialog.getExistingDirectory(self, "Select Source Labels Directory")
        if folder:
            self.src_lbls_dir = folder
            self.src_lbls_lbl.setText(folder)
            self.src_lbls_lbl.setStyleSheet("color: black;")

    def _select_out_dir(self):
        folder = QFileDialog.getExistingDirectory(self, "Select Output Directory")
        if folder:
            self.out_dir = folder
            self.out_dir_lbl.setText(folder)
            self.out_dir_lbl.setStyleSheet("color: black;")

    def _select_model(self):
        model, _ = QFileDialog.getOpenFileName(self, "Select YOLO Model", "", "YOLO Models (*.pt *.onnx)")
        if model:
            self.model_path = model
            self.model_lbl.setText(model)
            self.model_lbl.setStyleSheet("color: black;")

    def _toggle_search(self):
        if self.search_thread is not None and self.search_thread.isRunning():
            self.search_thread.stop()
            self.search_btn.setText("Search dataset")
            self.status_lbl.setText("Stopping...")
            return

        if not all([self.src_imgs_dir, self.src_lbls_dir, self.out_dir, self.model_path]):
            QMessageBox.warning(self, "Missing Configuration", "Please select all directories and model path.")
            return

        iou_threshold = self.iou_spinbox.value()
        min_instances = self.min_inst_spinbox.value()
        max_instances = self.max_inst_spinbox.value()

        if min_instances > max_instances:
            QMessageBox.warning(self, "Invalid Configuration", "Min Instances cannot be greater than Max Instances.")
            return

        self.search_btn.setText("Stop Search")
        self.progress_bar.setValue(0)
        self.status_lbl.setText("Starting search...")

        self.search_thread = SearchThread(
            img_dir=self.src_imgs_dir,
            lbl_dir=self.src_lbls_dir,
            out_dir=self.out_dir,
            model_path=self.model_path,
            iou_threshold=iou_threshold,
            min_instances=min_instances,
            max_instances=max_instances
        )

        self.search_thread.progress.connect(self._on_progress)
        self.search_thread.log.connect(self._on_log)
        self.search_thread.error.connect(self._on_error)
        self.search_thread.finished.connect(self._on_finished)
        self.search_thread.start()

    def _on_progress(self, current, total):
        self.progress_bar.setMaximum(total)
        self.progress_bar.setValue(current)
        self.status_lbl.setText(f"Processing... {current}/{total}")

    def _on_log(self, msg):
        self.status_lbl.setText(msg)

    def _on_error(self, err):
        QMessageBox.critical(self, "Search Error", err)
        self._on_finished()

    def _on_finished(self):
        self.search_btn.setText("Search dataset")
        self.status_lbl.setText("Finished.")
        self.search_thread = None

    def get_session_state(self) -> dict:
        return {
            "src_imgs_dir": self.src_imgs_dir,
            "src_lbls_dir": self.src_lbls_dir,
            "out_dir": self.out_dir,
            "model_path": self.model_path,
            "iou_threshold": self.iou_spinbox.value(),
            "min_instances": self.min_inst_spinbox.value(),
            "max_instances": self.max_inst_spinbox.value()
        }

    def restore_session_state(self, state: dict) -> list:
        warnings = []
        
        # Restore Img Dir
        src_imgs = state.get("src_imgs_dir", "")
        if src_imgs:
            if os.path.isdir(src_imgs):
                self.src_imgs_dir = src_imgs
                self.src_imgs_lbl.setText(src_imgs)
                self.src_imgs_lbl.setStyleSheet("color: black;")
            else:
                warnings.append(f"Find Dataset: Source Images folder not found - {src_imgs}")

        # Restore Lbls Dir
        src_lbls = state.get("src_lbls_dir", "")
        if src_lbls:
            if os.path.isdir(src_lbls):
                self.src_lbls_dir = src_lbls
                self.src_lbls_lbl.setText(src_lbls)
                self.src_lbls_lbl.setStyleSheet("color: black;")
            else:
                warnings.append(f"Find Dataset: Source Labels folder not found - {src_lbls}")

        # Restore Out Dir
        out_path = state.get("out_dir", "")
        if out_path:
            # Output might not exist yet if manually deleted, but we keep the path
            self.out_dir = out_path
            self.out_dir_lbl.setText(out_path)
            self.out_dir_lbl.setStyleSheet("color: black;")

        # Restore Model
        model = state.get("model_path", "")
        if model:
            if os.path.isfile(model):
                self.model_path = model
                self.model_lbl.setText(model)
                self.model_lbl.setStyleSheet("color: black;")
            else:
                warnings.append(f"Find Dataset: YOLO model not found - {model}")

        # Restore Thresholds
        self.iou_spinbox.setValue(state.get("iou_threshold", 0.50))
        self.min_inst_spinbox.setValue(state.get("min_instances", 0))
        self.max_inst_spinbox.setValue(state.get("max_instances", 1))

        return warnings
