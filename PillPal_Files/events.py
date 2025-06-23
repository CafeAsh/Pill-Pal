from enum import Enum
from enum import auto

class Events(Enum):
    # enums used as event identifiers
    START_SYSTEM = auto()
    DISPENSE_DOSE = auto()
    ADD_DOSE = auto()
    DELETE_DOSE = auto()
    SAVE_DOSE = auto()
    RESET_DOSES = auto()
    RESET_HISTORY = auto()
    
    def equals(self, event):
        return self == event
