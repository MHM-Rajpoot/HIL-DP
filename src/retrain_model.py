import os
import pandas as pd
import shutil
import joblib
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split

REF_PATH = "data/reference.csv"
CURR_PATH = "data/current.csv"
MODEL_OUT = "models/churn_model_v2.pkl"

df = pd.concat([
    pd.read_csv(REF_PATH),
    pd.read_csv(CURR_PATH)
])

X = df.drop(columns=["Churn"])
y = df["Churn"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

model = RandomForestClassifier(
    n_estimators=400,
    max_depth=10,
    random_state=42
)

model.fit(X_train, y_train)

os.makedirs(os.path.dirname(MODEL_OUT), exist_ok=True)
joblib.dump(model, MODEL_OUT)

# Update reference data with current data (new baseline after retraining)
print("📊 Updating reference data with current data as new baseline...")
shutil.copy(CURR_PATH, REF_PATH)

print("✅ Model retrained and saved")
print("✅ Reference data updated - drift will be 0% on next run")
