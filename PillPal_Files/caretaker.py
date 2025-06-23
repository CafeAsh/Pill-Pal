import json


class Caretaker(): 
    # class to keep track of pill data and schedules
    
    def __init__(self, mediator, data_file):
        # load data from data.json as a dict
        # initialize alarm
        self.mediator = mediator
        self.data_file = data_file
        with open(self.data_file, "r") as f:
            self.data = json.load(f)
        
    def notify(self):
        """alarm calls this to alert caretaker"""
        # handle notifications from alarm
    
    def get_pills(self, doseID):
        '''returns a list object'''
        return self.data.get("doses", {}).get(doseID, [])
        
    def load_doses(self):
        return self.data.get("doses", {})
        
    def load_history(self):
        return self.data.get("history", [])
    
    def add_dose(self, dose):
        doseID = dose.get("id", "")
        dose_data = dose.get("dose", [])
            
        self.data["doses"][doseID] = dose_data
        self.data["last_id"] = int(doseID)

        self.write_data()
            
    def get_next_id(self):
        next_id = self.data.get("last_id", 0) + 1
        self.data["last_id"] = next_id
        return next_id
            
    def delete_dose(self, doseID):
        self.data["doses"].pop(doseID, None)
        self.write_data()
            
    def save_dose(self, doseID, dose_data):
        self.data["doses"][doseID] = dose_data
        self.write_data()
            
    def reset_doses(self):
        self.data["doses"] = {}
        self.data["last_id"] = 0
        self.write_data()
        
    def reset_history(self):
        self.data["history"] = []
        self.write_data()
        
    def log_history_entry(self, msg, error):
        self.data.setdefault("history", []).append([msg, error])
        self.write_data()
        
    def write_data(self):
        with open(self.data_file, "w") as f: 
            json.dump(self.data, f, indent=4)
            
        
    # functions for mutating data dict and saving to data.json 

    
