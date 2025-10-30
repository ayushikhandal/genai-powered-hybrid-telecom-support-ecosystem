# This script saves synthetic data to a JSON file for the chatbot to use.
import json
from synthetic_data_generator import generate_synthetic_data

data = generate_synthetic_data(50)

with open("synthetic_data.json", "w") as f:
    json.dump(data, f, indent=2)

print("Synthetic data saved to synthetic_data.json")
