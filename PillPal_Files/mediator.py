# the main driver of the project. Manages interface between subsystems.
from events import Events
from verification import Verification
from servo_mgr import ServoManager
from caretaker import Caretaker
from kivy.clock import Clock
from threading import Event
import time
from datetime import datetime

class Mediator():
    
    def __init__(self, kivy_app):
        self.verification = Verification()
        self.servo_mgr = ServoManager()
        self.caretaker = Caretaker(self, "data.json")
        self.screen = kivy_app # reference to kivy app
        
        self.pill_dispensed_event = Event()
        self.confirm_event = Event()
        
    def display_msg(self, message):
        Clock.schedule_once(lambda dt: self.screen.display_msg(message))

    def notify(self, any_object: object, event):
        """handle events where a subsystem needs to initiate"""
        
        if Events.START_SYSTEM.equals(event):
            print("starting system")
            doses = self.caretaker.load_doses()
            if doses: 
                for doseID in sorted(doses.keys()): 
                    Clock.schedule_once(lambda dt, d=doseID: self.screen.add_dose(d))
                    
            history = self.caretaker.load_history()
            if history: 
                for entry in history: 
                    Clock.schedule_once(lambda dt, e=entry: self.screen.on_dose_complete({"history": e[0], "error": e[1]}))
        elif Events.DISPENSE_DOSE.equals(event):
            
            dose_data = self.caretaker.get_pills(any_object["doseID"])
                
            error = False
            retry = 0
            for pill_dose in dose_data:
                pill_type = pill_dose[0]
                dose_count = pill_dose[1]
                
                i = 0
                while i < dose_count:
                    if retry > 0: 
                        self.display_msg(f"RETRY {retry}/3: dispense {i + 1}/{dose_count} of {pill_type}")
                    else: 
                        self.display_msg(f"dispense {i + 1}/{dose_count} of {pill_type}")
                    
                    self.pill_dispensed_event.clear() # wait for dispense
                    self.pill_dispensed_event.wait() 
                    
                    if self.verification.detect_pill(pill_type):
                        i += 1  # only move forward if successful
                        retry = 0
                        Clock.schedule_once(lambda dt: self.screen.set_banner("green"))
                        self.display_msg("pill detected!")
                        self.servo_mgr.release_door(1)
                        time.sleep(1)
                        Clock.schedule_once(lambda dt: self.screen.set_banner("default"))
                    else:
                        retry += 1
                        Clock.schedule_once(lambda dt: self.screen.set_banner("red"))
                        self.display_msg("pill not detected")
                        time.sleep(1)
                        Clock.schedule_once(lambda dt: self.screen.set_banner("default"))
                        if retry == 3:
                            self.display_msg(f"failed to dispense {pill_type}, successfully dispensed {i}/{dose_count}")
                            error = True
                            Clock.schedule_once(lambda dt: self.screen.set_banner("red"))
                            Clock.schedule_once(lambda dt: self.screen.offer_confirm())
                            self.confirm_event.clear()
                            self.confirm_event.wait()
                            Clock.schedule_once(lambda dt: self.screen.unoffer_confirm())
                            Clock.schedule_once(lambda dt: self.screen.set_banner("default"))
                            retry = 0
                            break
            
            Clock.schedule_once(lambda dt: self.screen.set_banner("green"))
            self.display_msg("done!")
            Clock.schedule_once(lambda dt: self.screen.offer_confirm())
            self.confirm_event.clear()
            self.confirm_event.wait()
            Clock.schedule_once(lambda dt: self.screen.unoffer_confirm())
            Clock.schedule_once(lambda dt: self.screen.set_banner("default"))
            
            history_msg = f"Dose {any_object['doseID']} taken at {datetime.now().strftime('%a, %b %d %-I:%M %p')}"
            self.caretaker.log_history_entry(history_msg, error)
            
            Clock.schedule_once(lambda dt: self.screen.on_dose_complete({"history": history_msg, "error": error}))
            
            # TODO: RELEASE DOSE FROM HOLDING AREA
        elif Events.ADD_DOSE.equals(event):
            self.caretaker.add_dose(any_object)
        elif Events.DELETE_DOSE.equals(event):
            self.caretaker.delete_dose(any_object["doseID"])
        elif Events.SAVE_DOSE.equals(event):
            self.caretaker.save_dose(any_object["doseID"], any_object["dose_data"])
        elif Events.RESET_DOSES.equals(event):
            self.caretaker.reset_doses()
        elif Events.RESET_HISTORY.equals(event):
            self.caretaker.reset_history()
            
