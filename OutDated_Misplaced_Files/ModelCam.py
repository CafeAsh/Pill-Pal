from picamera2 import Picamera2
from models.common import DetectMultiBackend
from utils.general import (non_max_suppression, scale_boxes, check_img_size, cv2)
from utils.torch_utils import select_device
from utils.augmentations import letterbox
from ultralytics.utils.plotting import Annotator, colors
from pillmatch import load_pills_from_db, find_match

import torch
import numpy as np
import time
import RPi.GPIO as GPIO

# Configs
weights_path = 'multi2.pt'
imgsz = (640, 736)
conf_thres = 0.25
iou_thres = 0.45

# Initialize PiCamera2
picam2 = Picamera2()
config = picam2.create_video_configuration( main={"size": imgsz, "format" : "RGB888"} )
picam2.configure(config)
picam2.start()
time.sleep(2)

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
    SetAngle(60)
    time.sleep(0.5)

    SetAngle(0)

# Set device
device = select_device('')
model = DetectMultiBackend(weights_path, device=device)
stride, names, pt = model.stride, model.names, model.pt
imgsz = check_img_size(imgsz, s=stride)
model.warmup(imgsz=(1, 3, *imgsz))

#initialize database
pill_database = load_pills_from_db()

print(weights_path)
print(model.names)

print ("LED on")
GPIO.output(18, GPIO.HIGH)


count = 0
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
            descriptors = set()
            det[:, :4] = scale_boxes(im.shape[2:], det[:, :4], im0.shape).round()
            for *xyxy, conf, cls in reversed(det):
                label = f'{names[int(cls)]} {conf:.2f}'
                annotator.box_label(xyxy, label, color=colors(int(cls), True))
                descriptor_label = names[int(cls)].lower()
                descriptors.add(descriptor_label)
                
            matched_pills = find_match(pill_database, descriptors, threshold=0.75)
            print(descriptors, matched_pills)
             
            if(count>=3):
                if matched_pills:  
                    for pill in matched_pills:
                        print(descriptors)
                        print(f"Pill dispense: {pill}")
                        run_servo()
                        count=0                   
            
        else:
            #GPIO Control
            time.sleep(0.5) #DELAY

    cv2.imshow("PiCam YOLOv5", annotator.result())
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

print ("LED off")
GPIO.output(18, GPIO.LOW)
cv2.destroyAllWindows()
pwm.stop()
GPIO.cleanup()


            
            
