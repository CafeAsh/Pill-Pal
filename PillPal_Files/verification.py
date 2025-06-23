import time
import torch
import numpy as np
from picamera2 import Picamera2
from models.common import DetectMultiBackend
from utils.general import (non_max_suppression, scale_boxes, check_img_size, cv2)
from utils.torch_utils import select_device
from utils.augmentations import letterbox
from ultralytics.utils.plotting import Annotator, colors
from pillmatch import load_pills_from_db, find_match  # Added database logic
import gc

class Verification:
    def __init__(self):
        # Detection model setup
        self.model_paths = ['Pill.pt', 'Color.pt', 'Size.pt', 'Shape.pt']
        self.imgsz = (1216, 704)
        self.conf_thres = 0.25
        self.iou_thres = 0.45
        self.device = select_device('')
        self.models = []
        self.detected_labels = []

        for path in self.model_paths:
            model = DetectMultiBackend(path, device=self.device)
            model.warmup(imgsz=(1, 3, *self.imgsz))
            self.models.append((path, model))

        # Camera setup
        self.picam = Picamera2()
        self.picam.configure(self.picam.create_still_configuration())
        self.picam.start()
        time.sleep(2)
        
        self.picam.set_controls({
            "AwbEnable": False,                 # Turn off auto white balance
            "AnalogueGain": 2.0,               # Brightness gain (1.0–8.0)
            "ColourGains": (2.10546875, 1.2886472940444946),         # (Red gain, Blue gain)
            "ExposureTime": 27712              # In microseconds (adjust for brightness)
        })
        
        time.sleep(0.5)

        # Load pill database once at startup
        self.pill_database = load_pills_from_db()
        
    def append_to_file(descriptors, matched_pills):
        with open(filename, "a") as file:
            file.write(f"Characteristics: {descriptors}, Pill: {matched_pills}\n")
            
    def append_to_fail( descriptors, matched_pills):
        with open(filename, "a") as file:
            file.write(f"Failed Test: Characteristics: {descriptors}, Pill: {matched_pills}\n")

    def detect_pill(self, expected_pill_name, duration=5, required_detection_time=1, threshold=0.75):

        
        start_time = time.time()
        detection_start = None
        self.detected_labels.clear()
        filename = "50Zinc.txt"

        while time.time() - start_time < duration:
            frame = self.picam.capture_array()
            frame_bgr = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
            img0 = frame.copy()

            # Preprocess image
            img = letterbox(img0, self.imgsz, stride=32)[0]
            img = img.transpose((2, 0, 1))
            img = np.ascontiguousarray(img)
            img_tensor = torch.from_numpy(img).to(self.device).float() / 255.0
            if img_tensor.ndimension() == 3:
                img_tensor = img_tensor.unsqueeze(0)

            any_detected = False
            final_conf = -1

            for model_path, model in self.models:
                pred = model(img_tensor)[0]
                pred = non_max_suppression(pred, self.conf_thres, self.iou_thres)

                annotator = Annotator(img0.copy(), line_width=2, font_size = 50, pil=False)
                for det in pred:
                    if len(det):
                        any_detected = True
                        det[:, :4] = scale_boxes(img_tensor.shape[2:], det[:, :4], img0.shape).round()
                        for *xyxy, conf, cls in reversed(det):
                            label = f'{model.names[int(cls)]}'
                            self.detected_labels.append(label)
                            annotator.box_label(xyxy, f"{label} {conf:.2f}", color=colors(int(cls), True))
                            final_conf = max(final_conf, conf.item())

                result_img = annotator.result()
                out_img = cv2.cvtColor(result_img, cv2.COLOR_RGB2BGR)
                out_name = f"result_{model_path.replace('.pt', '')}.jpg"
                cv2.imwrite(out_name, out_img)

                del model
                gc.collect()

            if any_detected:

                descriptors = list(set(self.detected_labels))
                    
                if "pill" not in descriptors:
                        print("No 'Pill' class detected - abort")
                        with open(filename, "a") as file:
                            file.write(f"Failed Test: Characteristics: {descriptors}, Pill: {matched_pills}\n")
                        return False
                        
                filtered_descriptors = [d for d in descriptors if d != "pill"]
                    
                matched_pills = find_match(self.pill_database, filtered_descriptors, threshold=threshold)


                for pill in matched_pills:
                    if expected_pill_name.lower() in str(pill).lower():
                        print(f"Descriptors: {descriptors}")
                        print(f"Pill dispense: {pill}")
                        with open(filename, "a") as file:
                            file.write(f"Characteristics: {descriptors}, Pill: {matched_pills}\n")
                        return True
                                     
                print(f"No match found for descriptors: {descriptors}")
                with open(filename, "a") as file:
                    file.write(f"Failed Test: Characteristics: {descriptors}, Pill: {matched_pills}\n")
                return False
                
        print(expected_pill_name)
        print("Not detected")
        with open(filename, "a") as file:
            file.write(f"Failed Test: Characteristics: {descriptors}, Pill: {matched_pills}\n")
        return False

    def get_detected_labels(self):
        return self.detected_labels
