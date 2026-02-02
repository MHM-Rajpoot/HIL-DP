import pandas as pd
import numpy as np

RAW_PATH = "data/raw/telco.csv"
REF_PATH = "data/reference.csv"
CURR_PATH = "data/current.csv"

df = pd.read_csv(RAW_PATH)

df = df.drop(columns=["customerID"])

df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors="coerce")
df = df.dropna()

df["Churn"] = (df["Churn"] == "Yes").astype(int)

binary_cols = [
    "gender", "Partner", "Dependents", "PhoneService",
    "PaperlessBilling"
]

for col in binary_cols:
    df[col] = (df[col] == "Yes").astype(int)

df = pd.get_dummies(df, drop_first=True)

df = df.sort_values("tenure")

split_idx = int(len(df) * 0.7)
reference = df.iloc[:split_idx]
current = df.iloc[split_idx:]

reference.to_csv(REF_PATH, index=False)
current.to_csv(CURR_PATH, index=False)

print("✅ Reference and current datasets created")
