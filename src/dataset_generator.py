import json
import random
from pathlib import Path
from config import RAW_DATA_PATH

TEMPLATES = [
    # Battery
    ("The battery range is {battery_adj}, but the charging time is {charging_adj}.", 
     [("battery range", "battery_adj", "Battery"), ("charging time", "charging_adj", "Battery")]),
    
    # Engine & Mileage
    ("The engine power is {engine_adj} and fuel economy is {mileage_adj}.", 
     [("engine power", "engine_adj", "Engine"), ("fuel economy", "mileage_adj", "Mileage")]),
    
    # Safety & Comfort
    ("Cabin comfort is {comfort_adj}, though safety features feel {safety_adj}.", 
     [("cabin comfort", "comfort_adj", "Comfort"), ("safety features", "safety_adj", "Safety")]),
    
    # Infotainment & Price
    ("The touchscreen infotainment is {info_adj}, but the price is {price_adj}.", 
     [("touchscreen infotainment", "info_adj", "Infotainment"), ("price", "price_adj", "Price")]),
    
    # Service & Vehicle
    ("Dealer service was {service_adj}, making the overall vehicle experience {vehicle_adj}.", 
     [("dealer service", "service_adj", "Service"), ("vehicle experience", "vehicle_adj", "Vehicle")])
]

POS_ADJECTIVES = ["excellent", "impressive", "outstanding", "superb", "great", "smooth"]
NEG_ADJECTIVES = ["too long", "disappointing", "sluggish", "overpriced", "poor", "unreliable"]

def generate_dataset(num_samples: int = 200, output_path: Path = RAW_DATA_PATH):
    """
    Generates a synthetic dataset of automotive reviews with aspect span annotations.
    """
    output_path.parent.mkdir(parents=True, exist_ok=True)
    dataset = []
    
    for i in range(num_samples):
        template, aspect_configs = random.choice(TEMPLATES)
        fillers = {}
        annotations = []
        
        for term, placeholder, category in aspect_configs:
            is_pos = random.choice([True, False])
            adj = random.choice(POS_ADJECTIVES if is_pos else NEG_ADJECTIVES)
            fillers[placeholder] = adj
            annotations.append({
                "aspect": term,
                "category": category,
                "sentiment": "Positive" if adj in POS_ADJECTIVES else "Negative"
            })
            
        review_text = template.format(**fillers)
        dataset.append({
            "review_id": f"REV-{i+1:05d}",
            "text": review_text,
            "aspects": annotations
        })
        
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(dataset, f, indent=2)
        
    print(f"Generated {len(dataset)} synthetic automotive reviews at {output_path}")

if __name__ == "__main__":
    generate_dataset()