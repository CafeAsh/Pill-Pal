import RPi.GPIO as GPIO
import time


Servo_pin = 14
GPIO.setmode(GPIO.BCM)
GPIO.setup(Servo_pin, GPIO.OUT)
p = GPIO.PWM(Servo_pin, 30)
p.start(0)


def SetAngle(angle):
	duty = (angle / 36) + 5
	GPIO.output(Servo_pin, True)    # Set pin high
	p.ChangeDutyCycle(duty)         #Place PWM signal over high pin
	time.sleep(1)                 # wait for motion
	GPIO.output(Servo_pin, False)
	p.ChangeDutyCycle(0)

try:
    while True:
        angle = int(input("Enter value -90 -> 90: "))
        if -90 <= angle <= 90:
            SetAngle(angle)
        else:
            print("Error, beyond limits")

finally:
    SetAngle(0)
    p.stop()
    GPIO.cleanup()
