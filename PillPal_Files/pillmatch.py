

import sqlite3

DB_PATH = 'intro.db' 

def load_pills_from_db():
    """
    Load pill data from the SQLite database into memory.
    Returns a list of dictionaries, each containing pill name and descriptors.
    """
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    cursor.execute("SELECT id, name, color1, color2, shape, size FROM pills")
    rows = cursor.fetchall()
    conn.close()

    pills = []
    for row in rows:
        pill_id, name, color1, color2, shape, size = row
        descriptors = {color1.lower(), shape.lower(), size.lower()}
        if color2 and color2.strip():
            descriptors.add(color2.lower())
        pills.append({'id': pill_id, 'name': name, 'descriptors': descriptors})
    return pills

def has_missing_required_descriptors(pill_desc, detected_desc, required_types):
    
    required = {d for d in pill_desc if d in required_types}
    return not required.issubset(detected_desc)
    

def find_match(pills, detected_descriptors, threshold):
    """
    Compare detected descriptors to loaded pills and return matches.
    
    Args:
        pills (list): Output of load_pills_from_db()
        detected_descriptors (set): e.g., {'red', 'capsule', 'medium'}

    Returns:
        list of matching pill names
    """
    detected_descriptors = set(d.lower() for d in detected_descriptors)
    required_types = {'red', 'blue', 'white', 'beige'}
    strong_matches = []
    back_matches = []

    for pill in pills:
        pill_desc = pill['descriptors']
        match_score = len(detected_descriptors & pill_desc) / len(pill_desc)
        
        if match_score >= threshold:
            strong_matches.append(pill['name'])
            
        elif detected_descriptors <= pill_desc:
            if required_types & pill_desc and not has_missing_required_descriptors(pill_desc, detected_descriptors, required_types):
                back_matches.append(pill['name'])
            
    if strong_matches:
        return strong_matches
            
    if len(back_matches) == 1:
        return back_matches

    return []
    
def find_target(pills, detected_descriptors, target, threshold):
    """
    Compare detected descriptors to loaded pills and return matches.
    
    Args:
        pills (list): Output of load_pills_from_db()
        detected_descriptors (set): e.g., {'red', 'capsule', 'medium'}

    Returns:
        list of matching pill names
    """
    
    detected_descriptors = set(d.lower() for d in detected_descriptors) #Esnure descriptors are lower case
    target_name = target.lower()                                        #Target name is lower
    
    strong_matches = []
    back_matches = []
    
    target_pill = next((pill for pill in pills if pill['name'].lower() == target_name), None)
    
    if not target_pill:
        return []
        
    pill_desc = target_pill['descriptors']
    match_score = len(detected_descriptors & pill_desc) / len(pill_desc)
        
    if match_score >= threshold:
        strong_matches.append(target_pill['name'])
            
    return strong_matches
            
