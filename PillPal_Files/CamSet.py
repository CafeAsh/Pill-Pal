import cv2
from picamera2 import Picamera2
import time

# Initialize camera
picam = Picamera2()
picam.configure(picam.create_still_configuration())
picam.start()
time.sleep(2)

auto_settings = picam.capture_metadata()
print("auto Setting:")
print(auto_settings)

# Optional manual camera controls — tweak these
picam.set_controls({
    "AwbEnable": False,                 # Turn off auto white balance
    "AnalogueGain": 2.0,               # Brightness gain (1.0–8.0)
    "ColourGains": (2.10546875, 1.2886472940444946),         # (Red gain, Blue gain)
    "ExposureTime": 27712              # In microseconds (adjust for brightness)
})

time.sleep(0.5)

print("Press 'q' in the window to exit")

try:
    while True:
        frame = picam.capture_array()
        frame_bgr = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
        cv2.imshow("Camera Tuner", frame_bgr)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
finally:
    picam.stop()
    cv2.destroyAllWindows()
