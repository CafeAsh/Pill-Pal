from events import Events

class Verification:
    def __init__(self):
        return

    def detect_pill(self, duration=5, required_detection_time=2):
        detected = input("pill detected? (y/n)")
        if(detected == "y"):
            return True
        else:
            return False
