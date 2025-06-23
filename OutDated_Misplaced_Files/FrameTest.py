from picamera2 import Picamera2
from models.common import DetectMultiBackend
from utils.general import (non_max_suppression, scale_boxes, check_img_size, cv2)
from utils.torch_utils import select_device
from utils.augmentations import letterbox
from ultralytics.utils.plotting import Annotator, colors
import torch
import numpy as np
import time
import gc

# Initialize camera and capture image
picam = Picamera2()
picam.configure(picam.create_still_configuration())
picam.start()
time.sleep(0.5)
frame = picam.capture_array()
frame_bgr = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
picam.stop()

cv2.imwrite("capture.jpg", frame_bgr)

# Set up device and models
device = select_device('')
model_paths = ['Pill.pt', 'Color2.pt']  # Add as many as needed

# Preprocess image
img0 = frame.copy()
img = letterbox(img0, new_shape=640, stride=32)[0]
img = img.transpose((2, 0, 1))  # HWC to CHW
img = np.ascontiguousarray(img)
img_tensor = torch.from_numpy(img).to(device).float() / 255.0
if img_tensor.ndimension() == 3:
    img_tensor = img_tensor.unsqueeze(0)

# Run detection for each model
for model_path in model_paths:
    print(f"\n--- Running model: {model_path} ---")
    

    model = DetectMultiBackend(model_path, device=device)
    stride, names = model.stride, model.names
    pred = model(img_tensor)[0]
    pred = non_max_suppression(pred, conf_thres=0.25, iou_thres=0.45)

    # Annotate
    annotator = Annotator(img0.copy(), line_width=2)

    for det in pred:
        if len(det):
            det[:, :4] = scale_boxes(img_tensor.shape[2:], det[:, :4], img0.shape).round()
            for *xyxy, conf, cls in reversed(det):
                label = f'{names[int(cls)]} {conf:.2f}'
                print(f" - Class: {names[int(cls)]}, Confidence: {conf:.2f}")
                annotator.box_label(xyxy, label, color=colors(int(cls), True))
        else:
            print(" - No detections")

    result_img = annotator.result()
    out_img = cv2.cvtColor(result_img, cv2.COLOR_RGB2BGR)
    out_name = f"result_{model_path.replace('.pt', '')}.jpg"
    cv2.imwrite(out_name, out_img)
    
    del model
    gc.collect()


print("\n All Models Complete")
