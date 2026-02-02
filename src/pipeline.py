import json
import subprocess
import sys

DRIFT_THRESHOLD = 0.25

print("\n🚀 Starting Human-in-the-Loop Drift Pipeline")

subprocess.run([sys.executable, "src/detect_drift.py"], check=True)

with open("reports/drift_report.json") as f:
    report = json.load(f)

result = report["metrics"][0]["result"]
drift_ratio = result["number_of_drifted_columns"] / result["number_of_columns"]

print(f"\n📊 Drift ratio: {drift_ratio:.2%}")

if drift_ratio >= DRIFT_THRESHOLD:
    print("\n📄 Review drift report:")
    print("➡ reports/drift_report.html")

    decision = input("Approve retraining? (y/n): ").strip().lower()

    if decision == "y":
        subprocess.run([sys.executable, "src/retrain_model.py"], check=True)
        print("✅ Model updated after human approval")
    else:
        print("✋ Retraining rejected by human")
else:
    print("✅ Drift acceptable — no action")
