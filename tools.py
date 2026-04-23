import json
import os
import re

#Tool implementations for the medical logistics agent
INVENTORY_FILE = "inventory.json"

#First two were a template for me basically, courtesy of Randall, it helped out for me to write the rest
def calculate_shipping_cost(distance_km, weight_kg):
    base_rate = 0.5
    weight_factor = 0.2
    return round((distance_km * base_rate) + (weight_kg * weight_factor), 2)


def parse_shipping_input(input_str):
    distance, weight = map(float, input_str.split(","))
    return calculate_shipping_cost(distance, weight)

#Everything below is what I was able to set up using above code as a template
def _load_inventory():
    if not os.path.exists(INVENTORY_FILE):
        seed_inventory()
    with open(INVENTORY_FILE, "r") as f:
        return json.load(f)


def _save_inventory(data):
    with open(INVENTORY_FILE, "w") as f:
        json.dump(data, f, indent=2)


def seed_inventory():
    data = {
        "amoxicillin_500mg": 120,
        "ibuprofen_200mg": 300,
        "saline_bag_1l": 80,
        "syringe_10ml": 500,
        "n95_mask": 1000
    }
    _save_inventory(data)


def inventory_lookup(input_str):
    item, required = input_str.split(",")
    item = item.strip().lower().replace(" ", "_")
    required = int(required.strip())

    inventory = _load_inventory()
    available = int(inventory.get(item, 0))

    if available <= 0:
        return (
            f"{item}: out of stock. "
            "No units available."
        )

    if available >= required:
        return (
            f"{item}: {available} available. Request can be fulfilled for {required}. "
            "If you want to keep inventory unchanged, call Keep Inventory with: "
            f"{item},{required},yes"
        )

    return (
        f"{item}: only {available} available, requested {required}. "
        "If you want to keep inventory unchanged, call Keep Inventory with: "
        f"{item},{available},yes"
    )


def keep_inventory(input_str):
    item, amount, confirm = input_str.split(",")
    item = item.strip().lower().replace(" ", "_")
    amount = int(amount.strip())
    confirm = confirm.strip().lower()

    if confirm != "yes":
        return "No inventory action taken. Set confirm to yes to continue."

    inventory = _load_inventory()
    current = int(inventory.get(item, 0))

    if current < amount:
        return f"Request is {amount}. Current stock for {item} is {current}. Inventory kept unchanged."

    return f"Confirmed: keeping inventory unchanged for {item}. Current stock: {current}"


def parse_model_output(text):
    if text is None:
        return None

    text = str(text).strip()
    if text == "":
        return None

    try:
        return json.loads(text)
    except Exception:
        pass

    match = re.search(r"```(?:json)?\s*(.*?)\s*```", text, flags=re.IGNORECASE | re.DOTALL)
    if match:
        block = match.group(1).strip()
        try:
            return json.loads(block)
        except Exception:
            pass

    first_obj = re.search(r"\{.*\}", text, flags=re.DOTALL)
    if first_obj:
        try:
            return json.loads(first_obj.group(0))
        except Exception:
            return None

    return None