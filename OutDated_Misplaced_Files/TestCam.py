
#!/usr/bin/env python 3
from picamera2 import Picamera2
from models.common import DetectMultiBackend
from utils.general import (non_max_suppression, scale_boxes, check_img_size, cv2)
from utils.torch_utils import select_device
from utils.augmentations import letterbox
from ultralytics.utils.plotting import Annotator, colors

import torch
import numpy as np
import time
import RPi.GPIO as GPIO
import keyboard


# Configs
weights_path = 'FireWeights.pt'
imgsz = (640, 736)
conf_thres = 0.40
iou_thres = 0.45

# Initialize PiCamera2
picam2 = Picamera2()
config = picam2.create_video_configuration( main={"size": imgsz, "format" : "RGB888"} )
picam2.configure(config)
picam2.start()
time.sleep(1)

# Initialize Servo
GPIO_SER1 = 2
GPIO.setmode(GPIO.BCM)
GPIO.setup(2, GPIO.OUT)
GPIO.setup(18, GPIO.OUT)

pwm=GPIO.PWM(2, 50)

pwm.start(0)

def SetAngle(angle):
	duty = angle / 18 + 2
	GPIO.output(2, True)
	pwm.ChangeDutyCycle(duty)
	time.sleep(0.5)
	GPIO.output(2, False)
	pwm.ChangeDutyCycle(0)
    
def run_servo():
    #Servo Control
    SetAngle(70)

    time.sleep(0.5)

    SetAngle(0)
    
def append_to_file(conf, test_num):
    with open("50test.txt", "a") as file:
        file.write(f"Test: {test_num}, Confidence: {conf}\n")
        
def append_to_fail(conf, test_num):
    with open("50test.txt", "a") as file:
        file.write(f"Failed Test: {test_num}, Confidence: {conf}\n")


# Set device
device = select_device('')
model = DetectMultiBackend(weights_path, device=device)
stride, names, pt = model.stride, model.names, model.pt
imgsz = check_img_size(imgsz, s=stride)
model.warmup(imgsz=(1, 3, *imgsz))

print(weights_path)
print(model.names)

with open("test.txt", "w") as file:
    pass

test_num = 1
count = 0

print ("LED on")
GPIO.output(18, GPIO.HIGH)

# Detection loop
while True:
    frame = picam2.capture_array(wait=True)
    if frame.shape[2] == 4:
        frame = frame[:, :, :3]
    img = letterbox(frame, imgsz, stride=stride, auto=pt)[0]
    img = img.transpose((2, 0, 1))[::-1]  # HWC to CHW, BGR to RGB
    img = np.ascontiguousarray(img)

    im = torch.from_numpy(img).to(model.device)
    im = im.half() if model.fp16 else im.float()
    im /= 255
    im = im[None]  # add batch dim

    pred = model(im, augment=False, visualize=False)
    pred = non_max_suppression(pred, conf_thres, iou_thres)

    im0 = frame.copy()
    annotator = Annotator(im0, line_width=2, example=str(names))
    
    for det in pred:
        if len(det):
            count += 1
            print("pill detected")
            
            #BOX Drawing
            det[:, :4] = scale_boxes(im.shape[2:], det[:, :4], im0.shape).round()
            for *xyxy, conf, cls in reversed(det):
                label = f'{names[int(cls)]} {conf:.2f}'
            if(count==3):
                print("Pill Dispense")
                run_servo()
                append_to_file(conf, test_num)
                count=0                   
                test_num += 1
            
        else:
            if keyboard.is_pressed("a"):
                run_servo()
                append_to_fail(conf, test_num)
                print ("LED off")
                GPIO.output(18, GPIO.LOW)
                count=0
                test_num += 1
                break



cv2.destroyAllWindows()
pwm.stop()
GPIO.cleanup()


            
            
