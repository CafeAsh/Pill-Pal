'''
#ORIGIONAL DATA COLLECTION FILES, BELOW HAS ADJUSTMENTS FROM ASH, KEEP SCROLLING!!!

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
weights_path = 'Multi3.pt'
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
    time.sleep(0.75)

    SetAngle(0)

# Set device
device = select_device('')
model = DetectMultiBackend(weights_path, device=device)
stride, names, pt = model.stride, model.names, model.pt
imgsz = check_img_size(imgsz, s=stride)
model.warmup(imgsz=(1, 3, *imgsz))

def append_to_file(test_num, descriptors, matched_pills):
    with open("AshTest.txt", "a") as file:
        file.write(f"Test: {test_num}, Characteristics: {descriptors}, Pill: {matched_pills}\n")
        
def append_to_fail(test_num, descriptors, matched_pills):
    with open("AshTest.txt", "a") as file:
        file.write(f"Failed Test: {test_num}, Characteristics: {descriptors}, Pill: {matched_pills}\n")

#initialize database
pill_database = load_pills_from_db()

print(weights_path)
print(model.names)

print ("LED on")
GPIO.output(18, GPIO.HIGH)


test_num = 1
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
                        print(test_num)
                        append_to_file(test_num, descriptors, matched_pills)
                        run_servo()
                        test_num += 1
                        count=0
                else: 
                    if (count>10):
                        append_to_fail(test_num, descriptors, matched_pills)
                        run_servo()
                        print (test_num)
                        count=0
                        test_num += 1
            

        time.sleep(0.5) #DELAY

    cv2.imshow("PiCam YOLOv5", annotator.result())
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

print ("LED off")
GPIO.output(18, GPIO.LOW)
cv2.destroyAllWindows()
pwm.stop()
GPIO.cleanup()
'''

from picamera2 import Picamera2
from models.common import DetectMultiBackend
from utils.general import (non_max_suppression, scale_boxes, check_img_size, cv2)
from utils.torch_utils import select_device
from utils.augmentations import letterbox
from ultralytics.utils.plotting import Annotator, colors
from pillmatch import load_pills_from_db, find_match, find_target
from datetime import datetime

import torch
import numpy as np
import time
import RPi.GPIO as GPIO

# Configs
weights_path = 'Multi4s.pt'
imgsz = (640, 736)
conf_thres = 0.25
iou_thres = 0.45
filename = "50Cordyceps2.txt"

# Initialize PiCamera2
picam2 = Picamera2()
config = picam2.create_video_configuration( main={"size": imgsz, "format" : "RGB888"} )
picam2.configure(config)
picam2.start()
time.sleep(2)

# Initialize Servo
GPIO_Servo_top = 14
GPIO.setmode(GPIO.BCM)
GPIO.setup(GPIO_Servo_top, GPIO.OUT)

p = GPIO.PWM(GPIO_Servo_top, 30)    #SG90 we use have a weird frequency (From testing this works best)
p.start(0)

#Create Instance of File or append to exisiting:
with open(filename, "a") as file:
        file.write(f"Test Batch: {datetime.now()}\n") #title of batch with date and time

def SetAngle(ServoIO, angle):
	duty = (angle / 36) + 5
	GPIO.output(ServoIO, True)      # Set pin high
	p.ChangeDutyCycle(duty)         #Place PWM signal over high pin
	time.sleep(1)                   # wait for motion
	GPIO.output(ServoIO, False)
	p.ChangeDutyCycle(0)
    
def run_servo(ServoIO):
    #Servo Control
    SetAngle(ServoIO, 60)
    time.sleep(0.75)
    SetAngle(ServoIO, 0)

# Set device
device = select_device('')
model = DetectMultiBackend(weights_path, device=device)
stride, names, pt = model.stride, model.names, model.pt
imgsz = check_img_size(imgsz, s=stride)
model.warmup(imgsz=(1, 3, *imgsz))

#Opens file to append success
def append_to_file(test_num, descriptors, matched_pills):
    with open(filename, "a") as file:
        file.write(f" Characteristics: {descriptors}, Pill: {matched_pills}\n")

#Opens file to append failure 
def append_to_fail(test_num, descriptors, matched_pills):
    with open(filename, "a") as file:
        file.write(f"Failed Test {test_num}, Characteristics: {descriptors}, Pill: {matched_pills}\n")

#initialize database
pill_database = load_pills_from_db()

print(weights_path)
print(model.names)

test_num = 1
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
            #print("pill detected")
            
            #BOX Drawing
            descriptors = set()
            det[:, :4] = scale_boxes(im.shape[2:], det[:, :4], im0.shape).round()
            for *xyxy, conf, cls in reversed(det):
                label = f'{names[int(cls)]} {conf:.2f}'
                annotator.box_label(xyxy, label, color=colors(int(cls), True))
                descriptor_label = names[int(cls)].lower()
                descriptors.add(descriptor_label)
                
            matched_pills = find_match(pill_database, descriptors, 0.75)
            #matched_pills = find_target(pill_database, descriptors, 'Cordyceps',0.75)
            print(descriptors, matched_pills)
             
            if(count>=3):
                if matched_pills:  
                    for pill in matched_pills:
                        print(descriptors)
                        print(f"Pill dispense: {pill}")
                        print(test_num)
                        append_to_file(test_num, descriptors, matched_pills)
                        run_servo(GPIO_Servo_top)
                        test_num += 1
                        count=0
                        
                        ''' Issue with find_match!! For something like the Cordyceps pill we get a detection when we have
                        beige and capsule, but we get a fail when we get beige, pill, and capsule. Idk what the issue is 
                        mainly cause of lack of understanding of the function. If you can try to iron this out that be rad.
                        find_target was my attempt at fixing it because in it real use case we would know the name of pill that
                        should be dispensed, thus we can just check its decriptor rather than loop through all the pills. Its 
                        not scaleable otherwise if we need to loop through 20000 pills in the db.'''
                        
                else: 
                    if (count>10):
                        append_to_fail(test_num, descriptors, matched_pills)
                        run_servo(GPIO_Servo_top)
                        print (test_num)
                        count=0
                        test_num += 1
            

        time.sleep(0.5) #DELAY
    
    cv2.imshow("PiCam YOLOv5", annotator.result())
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

#cv2.destroyAllWindows()
pwm.stop()
GPIO.cleanup()


            
            


            
            
