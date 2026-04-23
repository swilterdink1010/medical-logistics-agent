import json
import os
#Use this for data persistence this is what Randall had done
FILE = "memory.json"

def load_memory():
    if not os.path.exists(FILE):
        return []
    with open(FILE, "r") as f:
        return json.load(f)

def save_memory(data):
    with open(FILE, "w") as f:
        json.dump(data, f)