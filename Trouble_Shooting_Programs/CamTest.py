from picamera2 import Picamera2, Preview
import cv2
import time

picam2 = Picamera2() #Camera Instance

#Set Resolution
camera_config = picam2.create_preview_configuration(main={"size": (1000, 1000)})
picam2.configure(camera_config)

#picam2.start_preview(Preview.QTGL)
#Start Camera
picam2.start()
time.sleep(2)

while True:
	frame = picam2.capture_array()
	frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
	cv2.imshow("Pi Cam Feed", frame)
	
	if cv2.waitKey(1) & 0xFF == ord('q'):
		break
		
cv2.destroyAllWindows()
#sleep(60)
picam2.close()
