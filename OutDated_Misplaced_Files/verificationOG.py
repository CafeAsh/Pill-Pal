import time
import torch
import numpy as np
from picamera2 import Picamera2
from models.common import DetectMultiBackend
from utils.general import (non_max_suppression, scale_boxes, check_img_size, cv2)
from utils.torch_utils import select_device
from utils.augmentations import letterbox
from ultralytics.utils.plotting import Annotator, colors
from events import Events
import RPi.GPIO as GPIO

class Verification:
    def __init__(self):
        
        # Lighting LED 
        GPIO.setmode(GPIO.BCM)
        GPIO.setup(18, GPIO.OUT)

        # Model and image configuration
        self.weights_path = 'FireWeights.pt'
        self.imgsz = (640, 736)
        self.conf_thres = 0.50
        self.iou_thres = 0.45

        # Initialize YOLOv5 model
        self.device = select_device('')
        self.model = DetectMultiBackend(self.weights_path, device=self.device)
        self.stride, self.names, self.pt = self.model.stride, self.model.names, self.model.pt
        self.imgsz = check_img_size(self.imgsz, s=self.stride)
        self.model.warmup(imgsz=(1, 3, *self.imgsz))

        # Initialize PiCamera2
        self.picam2 = Picamera2()
        config = self.picam2.create_video_configuration(main={"size": self.imgsz, "format": "RGB888"})
        self.picam2.configure(config)
        self.picam2.start()
        time.sleep(2)

    def detect_pill(self, duration=3, required_detection_time=1):
        start_time = time.time()
        detection_start = None

        GPIO.output(18, GPIO.HIGH)
        time.sleep(0.5)
            
        while time.time() - start_time < duration:
            frame = self.picam2.capture_array(wait=True)
            if frame.shape[2] == 4:
                frame = frame[:, :, :3]

            # Preprocess image
            img = letterbox(frame, self.imgsz, stride=self.stride, auto=self.pt)[0]
            img = img.transpose((2, 0, 1))[::-1]
            img = np.ascontiguousarray(img)

            im = torch.from_numpy(img).to(self.model.device)
            im = im.half() if self.model.fp16 else im.float()
            im /= 255
            im = im[None]
            

            # Inference
            pred = self.model(im, augment=False, visualize=False)
            pred = non_max_suppression(pred, self.conf_thres, self.iou_thres)

            detected = any(len(det) > 0 for det in pred)
            
            #TESTING: BOX Drawing
            im0 = frame.copy()    
            annotator = Annotator(im0, line_width=2, example=str(self.names))

            for det in pred:
                det[:, :4] = scale_boxes(im.shape[2:], det[:, :4], im0.shape).round()
                for *xyxy, conf, cls in reversed(det):
                    label = f'{self.names[int(cls)]} {conf:.2f}'
                    annotator.box_label(xyxy, label, color=colors(int(cls), True))

            if detected:
                if detection_start is None:
                    detection_start = time.time()
                elif time.time() - detection_start >= required_detection_time:
                    print(f"detected with confidence {conf}")
                    GPIO.output(18, GPIO.LOW)
                    return True
            else:
                conf = -1
                detection_start = None
                
        GPIO.output(18, GPIO.LOW)
        
        print(f"not detected, conf {conf}")
        return False
