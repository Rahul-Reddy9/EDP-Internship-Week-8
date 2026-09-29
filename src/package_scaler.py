import os
import joblib
import pandas as pd
from sklearn.preprocessing import StandardScaler

RAW_PATH = os.path.join("data", "raw", "creditcard.csv")
MODEL_DIR = "models"

df = pd.read_csv(RAW_PATH)

scaler = StandardScaler()
scaler.fit(df[["Time", "Amount"]])

os.makedirs(MODEL_DIR, exist_ok=True)
joblib.dump(scaler, os.path.join(MODEL_DIR, "scaler.pkl"))

print("Scaler saved successfully to models/scaler.pkl")
