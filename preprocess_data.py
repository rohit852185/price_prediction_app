import pandas as pd
import os

# Get current file directory
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# Correct file paths
input_path = os.path.join(BASE_DIR, "train_original.csv")
output_path = os.path.join(BASE_DIR, "train_updated.csv")

# Load original dataset
df = pd.read_csv(input_path)

def update_price(row):
    year = row["Launch Year"]
    price = row["Price ($)"]

    if year <= 2019:
        return price * 0.15
    elif year == 2020:
        return price * 0.25
    elif year == 2021:
        return price * 0.35
    elif year == 2022:
        return price * 0.42     
    elif year == 2023:
        return price * 0.50
    else:
        return price * 0.55

df["Price ($)"] = df.apply(update_price, axis=1).astype(int)

# Save updated dataset
df.to_csv(output_path, index=False)

print("Updated dataset created successfully")