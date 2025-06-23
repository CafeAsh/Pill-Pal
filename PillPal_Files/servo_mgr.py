
import RPi.GPIO as GPIO
from time import sleep

class ServoManager:
    def __init__(self):
        # BCM pin numbers
        self.servo_pins = {
            1: 14,  # Door 1 → GPIO 2
            2: 3   # Door 2 → GPIO 3
        }

        GPIO.setmode(GPIO.BCM)
        GPIO.setup(14, GPIO.OUT)
        GPIO.setup(3, GPIO.OUT)
        
        self.pwm = [GPIO.PWM(14, 30), GPIO.PWM(3, 30)]
            
        for p in self.pwm: 
            p.start(0)

    def SetAngle(self, door, angle):
        pin = self.servo_pins[door]
        duty = angle / 36 + 5
        GPIO.output(pin, True)
        self.pwm[door-1].ChangeDutyCycle(duty)
        sleep(1)
        GPIO.output(pin, False)
        self.pwm[door-1].ChangeDutyCycle(0)

    def release_door(self, door):
        # Open and close the door using its servo
        print(f"release door {door}")
        if door in self.servo_pins:
            self.SetAngle(door, 60)  # Open
            sleep(1)
            self.SetAngle(door, 0)   # Close
        else:
            print(f"Invalid door ID: {door}")



