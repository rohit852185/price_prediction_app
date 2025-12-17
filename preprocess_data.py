import pandas as pd

# Load original dataset
df = pd.read_csv("train_original.csv")

def update_price(row):
    year = row["Launch Year"]
    price = row["Price ($)"]

    if year <= 2019:
        return price * 0.45
    elif year == 2020:
        return price * 0.55
    elif year == 2021:
        return price * 0.65
    elif year == 2022:
        return price * 0.72
    elif year == 2023:
        return price * 0.80
    else:
        return price * 0.85

df["Price ($)"] = df.apply(update_price, axis=1).astype(int)

# Save updated dataset
df.to_csv("train_updated.csv", index=False)

print("Updated dataset created successfully")