import os
import pandas as pd
import joblib
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_auc_score

DATA_PATH = "data/reference.csv"
MODEL_PATH = "models/churn_model_v1.pkl"

os.makedirs(os.path.dirname(MODEL_PATH), exist_ok=True)

df = pd.read_csv(DATA_PATH)

X = df.drop(columns=["Churn"])
y = df["Churn"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

model = RandomForestClassifier(
    n_estimators=300,
    max_depth=8,
    random_state=42
)

model.fit(X_train, y_train)

auc = roc_auc_score(y_test, model.predict_proba(X_test)[:, 1])
joblib.dump(model, MODEL_PATH)

print(f"✅ Model trained and saved to {MODEL_PATH} | ROC-AUC: {auc:.3f}")
