import os

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
KAGGLE_DIR = os.path.join(PROJECT_ROOT, "secrets")

os.environ["KAGGLE_CONFIG_DIR"] = KAGGLE_DIR

os.environ.pop("KAGGLE_USERNAME", None)
os.environ.pop("KAGGLE_KEY", None)

# ------------------------------------------------------------------------

from kaggle.api.kaggle_api_extended import KaggleApi
import pandas as pd

kaggle_json = os.path.join(KAGGLE_DIR, "kaggle.json")
if not os.path.exists(kaggle_json):
    raise FileNotFoundError(f"Missing kaggle.json at {kaggle_json}")

api = KaggleApi()
api.authenticate()

print("⬇ Downloading Telco Customer Churn dataset...")

os.makedirs("data/raw", exist_ok=True)

api.dataset_download_files(
    "blastchar/telco-customer-churn",
    path="data/raw",
    unzip=True
)

df = pd.read_csv("data/raw/WA_Fn-UseC_-Telco-Customer-Churn.csv")
df.to_csv("data/raw/telco.csv", index=False)

print("✅ Dataset downloaded using project-local kaggle.json")
