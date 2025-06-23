from kivy.app import App
from kivy.lang import Builder
from kivy.uix.label import Label
from kivy.uix.widget import Widget
from kivy.uix.button import Button
from kivy.uix.boxlayout import BoxLayout
from kivy.uix.screenmanager import Screen, ScreenManager, NoTransition
from kivy.properties import ListProperty, ObjectProperty, NumericProperty, StringProperty
from kivy.graphics import Color, Rectangle
from kivy.uix.spinner import Spinner
from kivy.clock import Clock
from kivy.core.window import Window
from mediator import Mediator
from events import Events
from datetime import datetime
import threading


class ColoredScreen(Screen):
    # class that supports colored backgrounds
    pass
class MyLabel(Label):
    # Custom label styling
    pass
class MyButton(Button):
    # Custom button styling
    pass
class DemoButton(MyButton):
    # Special button used in the pill dispensing demo screen
    pass
class PillRow(BoxLayout):
    # Widget representing a single pill type and quantity in a dose
    pill_type = StringProperty("Select Pill")
    quantity = NumericProperty(1)
    
    def decrement(self): # Decrease pill quantity by 1
        p = self.ids.pill_qty
        if p.text == "1":
            self.parent.remove_widget(self)
        p.text = str(max(1, int(p.text)-1))
        self.quantity = int(p.text)
        
    def increment(self): # Increase pill quantity by 1
        p = self.ids.pill_qty
        p.text = str(int(p.text)+1)
        self.quantity = int(p.text)
        
    pass
class ClockLabel(MyLabel):

    # Live updating clock display label

    time_text = StringProperty() # Current time string
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        # Schedule time update every second
        Clock.schedule_interval(self.update_time, 1)
        self.update_time(0)
        
    def update_time(self, dt): # Update the displayed time (called every second)
        now = datetime.now()
        self.time_text = f"Current Time: {now.strftime('%a, %b %d %I:%M %p')}"
    

class MainScreen(ColoredScreen):
    # Navigate to specific screens within device
    def start_demo(self):
        self.manager.current = "choose_dose"
    def to_history(self):
        self.manager.current = "history"
    def to_dose_mgr(self):
        self.manager.current = "dose_mgr"
        
class ChooseDoseScreen(ColoredScreen):
    
    mediator = ObjectProperty(None)
    
    def dispense_dose(self, doseID):
        self.manager.current = "dose_demo"
        threading.Thread(target=lambda: self.mediator.notify({"doseID": str(doseID)}, Events.DISPENSE_DOSE), daemon=True).start()
    def back(self):
        self.manager.current = "main"
        
class DoseDemoScreen(ColoredScreen):
    
    mediator = ObjectProperty(None)
    
    def offer_confirm(self):
        # 0.078, 0.588, 0.498, 1
        confirm_button = DemoButton(text="CONFIRM", on_press=lambda dt: self.confirm())
        layout = self.ids.demo_layout
        layout.remove_widget(layout.children[0])
        layout.add_widget(confirm_button, index=0)
        
    def unoffer_confirm(self):
        verify_button = DemoButton(text="VERIFY", on_press=lambda dt: self.verify())
        layout = self.ids.demo_layout
        layout.remove_widget(layout.children[0])
        layout.add_widget(verify_button, index=0)
        
    def verify(self):
        self.ids.banner.text="verifying pill..."
        self.mediator.pill_dispensed_event.set()
        
    def confirm(self):
        self.mediator.confirm_event.set()
        
    def back(self):
        self.manager.current = "choose_dose"
        
class HistoryScreen(ColoredScreen):
    
    mediator = ObjectProperty(None)
    
    def back(self):
        self.manager.current = "main"
        
    def reset_history(self):
        layout = self.ids.label_container
        for widget in layout.children[:-1]:
            layout.remove_widget(widget)
        threading.Thread(target=lambda: self.mediator.notify(None, Events.RESET_HISTORY), daemon=True).start()
        
class DoseMgrScreen(ColoredScreen):
    
    mediator = ObjectProperty(None)
    
    def back(self):
        self.manager.current = "main"
        
    def to_edit_dose(self, dose_id):
        self.manager.current = "edit_dose"
        self.manager.get_screen("edit_dose").load_screen(dose_id)
        
    def reset_doses(self): 
        threading.Thread(target=lambda: self.mediator.notify(None, Events.RESET_DOSES), daemon=True).start()
        
        cd_container = self.manager.get_screen("choose_dose").ids.dose_choice_container
        dm_container = self.manager.get_screen("dose_mgr").ids.dose_container
        
        for widget in dm_container.children[1:-1]:
            dm_container.remove_widget(widget)
        cd_container.clear_widgets()
        
    def add_dose(self):
        dose_id = self.mediator.caretaker.get_next_id()
        doseID = str(dose_id)
                
        new_dose = MyButton(text=f"Dose {doseID}")
        new_dose.bind(on_press = lambda dt: self.to_edit_dose(dose_id))
        self.ids.dose_container.add_widget(new_dose, -1)
        
        new_dose = MyButton(text=f"Dose {doseID}")
        new_dose.bind(on_press = lambda dt: self.manager.get_screen("choose_dose").dispense_dose(dose_id))
        self.manager.get_screen("choose_dose").ids.dose_choice_container.add_widget(new_dose)
        
        threading.Thread(target=lambda: self.mediator.notify({"id": doseID, "dose": []}, Events.ADD_DOSE), daemon=True).start()
        
        self.ids.scrollview.scroll_y = 1
        
class EditDoseScreen(ColoredScreen):
    
    mediator = ObjectProperty(None)
    doseID = StringProperty("default")
    pill_rows = NumericProperty(0)
    
    def back(self):
        self.manager.current = "dose_mgr"
        self.ids.pill_rows.clear_widgets()
        
    def add_pill_row(self):
        if self.pill_rows >= 10:
            return
        self.ids.pill_rows.add_widget(PillRow())
        self.pill_rows += 1
        
    def load_screen(self, dose_id):
        self.doseID = str(dose_id)
        self.ids.header.text = f"Editing Dose {self.doseID}"
        dose_data = self.mediator.caretaker.get_pills(self.doseID)
        
        for dose in dose_data: 
            pill_type = str(dose[0])
            quantity = dose[1]
            
            row = PillRow(pill_type=pill_type, quantity=quantity)
            self.ids.pill_rows.add_widget(row)
            
    def save_dose(self):
        dose_data = []
        for row in reversed(self.ids.pill_rows.children): 
            pill = row.pill_type
            qty = row.quantity
            
            if pill == "Select Pill" or not pill: 
                continue
                
            dose_data.append([pill,qty])
        threading.Thread(target=lambda: self.mediator.notify({"doseID": self.doseID, "dose_data": dose_data}, Events.SAVE_DOSE), daemon=True).start() 
        self.ids.pill_rows.clear_widgets()
        self.manager.current = "dose_mgr"           
            
    def delete_dose(self):
        # always called after load_screen
        threading.Thread(target=lambda: self.mediator.notify({"doseID": self.doseID}, Events.DELETE_DOSE), daemon=True).start()
        
        choose_dose_container = self.manager.get_screen("choose_dose").ids.dose_choice_container
        dose_mgr_container = self.manager.get_screen("dose_mgr").ids.dose_container
        
        for widget in choose_dose_container.children[:]:
            if isinstance(widget, Button) and widget.text == f"Dose {self.doseID}":
                choose_dose_container.remove_widget(widget)
                break
                
        for widget in dose_mgr_container.children[:]:
            if isinstance(widget, Button) and widget.text == f"Dose {self.doseID}":
                dose_mgr_container.remove_widget(widget)
                break
                
        self.back()

class MyApp(App):

    def build(self):
        mediator = Mediator(self)
        sm = ScreenManager()
        sm.transition = NoTransition()
        
        ## screens ##
        main_screen = MainScreen(name="main")
        
        choose_dose_screen = ChooseDoseScreen(name="choose_dose")
        choose_dose_screen.mediator = mediator
        
        dose_demo_screen = DoseDemoScreen(name="dose_demo")
        dose_demo_screen.mediator = mediator
        
        history_screen = HistoryScreen(name="history")
        history_screen.mediator = mediator
        
        dose_mgr_screen = DoseMgrScreen(name="dose_mgr")
        dose_mgr_screen.mediator = mediator
        
        edit_dose_screen = EditDoseScreen(name="edit_dose")
        edit_dose_screen.mediator = mediator
        
        ## add screens to screen manager ##
        sm.add_widget(main_screen)
        sm.add_widget(choose_dose_screen)
        sm.add_widget(dose_demo_screen)
        sm.add_widget(history_screen)
        sm.add_widget(dose_mgr_screen)
        sm.add_widget(edit_dose_screen)
        
        mediator.notify(None, Events.START_SYSTEM)
        return sm
        
    def display_msg(self, msg):
        dose_demo_screen = self.root.get_screen("dose_demo")
        dose_demo_screen.ids.banner.text = msg
        
    def set_banner(self, color):
        dose_demo_screen = self.root.get_screen("dose_demo")
        banner = dose_demo_screen.ids.banner
        
        set_banner_color(banner, color)
        
    def offer_confirm(self):
        dose_demo_screen = self.root.get_screen("dose_demo")
        dose_demo_screen.offer_confirm()
        
    def unoffer_confirm(self):
        dose_demo_screen = self.root.get_screen("dose_demo")
        dose_demo_screen.unoffer_confirm()
        
    def on_dose_complete(self, data):
        self.root.current = "main"
        history = self.root.get_screen("history")
        
        new_entry = MyLabel(text=data["history"])
        if data["error"] == True:
            set_banner_color(new_entry, "red")
        
        history.ids.label_container.add_widget(new_entry, -1)
        
    def add_dose(self, doseID):
        # used to load ui from data.json
        dose_mgr_screen = self.root.get_screen("dose_mgr")
        choose_dose_screen = self.root.get_screen("choose_dose")
        
        new_dose = MyButton(text=f"Dose {doseID}")
        new_dose.bind(on_press = lambda dt: dose_mgr_screen.to_edit_dose(doseID))
        dose_mgr_screen.ids.dose_container.add_widget(new_dose, -1)
        
        new_dose = MyButton(text=f"Dose {doseID}")
        new_dose.bind(on_press = lambda dt: choose_dose_screen.dispense_dose(doseID))
        choose_dose_screen.ids.dose_choice_container.add_widget(new_dose)

def set_banner_color(banner, color):
    instrs = banner.canvas.before.children
        
    for instr in instrs:
        if (isinstance(instr, Color)):
            if(color == "red"):
                instr.rgba = (1,0.41,0.35,1)
            elif(color == "green"):
                instr.rgba = (0.886,0.988, 0.839, 1) #(0.078, 0.588, 0.498, 1) #(0.88,0.98,0.84,1)
            elif(color == "default"):
                instr.rgba = (0.8,0.925,0.933,1)
            break
        
        
if __name__ == "__main__":
    
    Window.size = (500, 300)
    myapp = MyApp()
    myapp.run()
    
