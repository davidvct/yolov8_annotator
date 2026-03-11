from ultralytics import YOLO
import os

model_path = r"L:\DataScience_L\ShuenWei\temp\qe-py-yolov8-drone-module-detection-main\runs\segment\train10\weights\best.pt"
output_name = "L:\DataScience_L\ShuenWei\temp\qe-py-yolov8-drone-module-detection-main\runs\segment\train10\weights\best_custom_export.onnx"

# --- Configuration ---
# Precision Options: 'fp32', 'fp16', 'int8'
# - 'fp32': Standard 32-bit floating point (Default)
# - 'fp16': Half-precision 16-bit floating point (Smaller, can be faster on GPU)
# - 'int8': 8-bit Integer Quantization (Smallest, fastest on CPU, requires dataset for calibration)
EXPORT_PRECISION = 'fp32' 

# Other Options
SIMPLIFY = True  # Use onnx-simplifier to clean up the graph (Recommended)
DYNAMIC = False  # Use dynamic input shapes (e.g. variable batch size)
OPSET = 12       # ONNX Opset version
# ---------------------

print(f"Loading model from {model_path}...")
model = YOLO(model_path)

print(f"Exporting to ONNX with precision: {EXPORT_PRECISION}...")

# Prepare arguments
kwargs = {
    'format': 'onnx',
    'imgsz': 640,
    'opset': OPSET,
    'simplify': SIMPLIFY,
    'dynamic': DYNAMIC
}

# Apply Precision Settings
if EXPORT_PRECISION == 'fp16':
    kwargs['half'] = True
    kwargs['int8'] = False
elif EXPORT_PRECISION == 'int8':
    kwargs['int8'] = True
    kwargs['half'] = False
    # Note: INT8 export usually requires a dataset for calibration. 
    # Unless specified, it might default to COCO128 or require a 'data' argument.
    # kwargs['data'] = 'path/to/data.yaml' 
else: # fp32
    kwargs['half'] = False
    kwargs['int8'] = False
    
# Export the model
path = model.export(**kwargs)

print(f"Export complete: {path}")
