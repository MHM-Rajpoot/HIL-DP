import os
import sys
import webbrowser
import base64
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from evidently.report import Report
from evidently.metrics import DatasetDriftMetric, ColumnDriftMetric

# Check if running in headless mode (from web pipeline)
HEADLESS = '--no-browser' in sys.argv
# Check if showing all columns (after approval)
SHOW_ALL_COLUMNS = '--show-all' in sys.argv

# Set style for better visuals
plt.style.use('seaborn-v0_8-darkgrid')
sns.set_palette("husl")

REF_PATH = "data/reference.csv"
CURR_PATH = "data/current.csv"
REPORT_DIR = "reports"

HTML_REPORT = os.path.join(REPORT_DIR, "drift_report.html")
JSON_REPORT = os.path.join(REPORT_DIR, "drift_report.json")
VISUAL_REPORT = os.path.join(REPORT_DIR, "drift_visualizations.png")

os.makedirs(REPORT_DIR, exist_ok=True)

reference = pd.read_csv(REF_PATH)
current = pd.read_csv(CURR_PATH)

# Get all column names for per-column drift detection
all_columns = reference.columns.tolist()

# Create metrics list with dataset drift + column drift for each column
metrics_list = [DatasetDriftMetric()]
for col in all_columns:
    metrics_list.append(ColumnDriftMetric(column_name=col))

report = Report(metrics=metrics_list)

report.run(
    reference_data=reference,
    current_data=current
)

report.save_html(HTML_REPORT)
report.save_json(JSON_REPORT)

result = report.as_dict()
metric = result["metrics"][0]["result"]

# Extract per-column drift results
column_drift_results = {}
for m in result["metrics"][1:]:  # Skip first (DatasetDriftMetric)
    col_name = m["result"].get("column_name", "")
    if col_name:
        column_drift_results[col_name] = {
            "drift_detected": m["result"].get("drift_detected", False),
            "drift_score": m["result"].get("drift_score", 0),
            "stattest_name": m["result"].get("stattest_name", "")
        }

# Get list of drifted columns
drifted_columns = [col for col, info in column_drift_results.items() if info["drift_detected"]]
print(f"📊 Drifted columns ({len(drifted_columns)}): {drifted_columns}")

# -----------------------------
# Visual Report Generation
# -----------------------------
print("\n" + "="*50)
print("📊 DRIFT DETECTION VISUAL REPORT")
print("="*50)

# Get numeric columns for distribution plots (exclude zero-variance columns)
numeric_cols = reference.select_dtypes(include=['number']).columns.tolist()
# Filter out columns with zero or near-zero variance
valid_cols = [col for col in numeric_cols if reference[col].std() > 0.01 and current[col].std() > 0.01]

# Prioritize drifted columns that have valid variance
drifted_valid = [col for col in drifted_columns if col in valid_cols]
non_drifted_valid = [col for col in valid_cols if col not in drifted_columns]

# Number of drifted columns to show
num_drifted = len(drifted_valid)
print(f"📈 Found {num_drifted} drifted columns with valid variance for plotting")
print(f"📊 Total valid columns available: {len(valid_cols)}")
print(f"🔧 SHOW_ALL_COLUMNS flag: {SHOW_ALL_COLUMNS}")

# Determine which columns to plot - all valid columns if --show-all, otherwise just drifted
if SHOW_ALL_COLUMNS:
    plot_cols = valid_cols  # Show ALL valid columns when --show-all is set
    print(f"📊 Showing ALL {len(plot_cols)} columns with valid variance")
else:
    plot_cols = drifted_valid[:12] if drifted_valid else valid_cols[:3]
    print(f"📊 Showing {len(plot_cols)} columns")

# Layout: All rows have 3 columns
num_dist_plots = len(plot_cols)
dist_rows = (num_dist_plots + 2) // 3  # Number of rows for distribution plots (3 per row)
total_rows = 1 + dist_rows  # 1 row for summary + rows for distributions

# Create figure with dynamic size
fig_height = 4 * total_rows
fig = plt.figure(figsize=(15, fig_height))
fig.suptitle('Data Drift Analysis Dashboard', fontsize=18, fontweight='bold', y=0.98)

# Use GridSpec for uniform 3-column layout
from matplotlib.gridspec import GridSpec
gs = GridSpec(total_rows, 3, figure=fig, hspace=0.35, wspace=0.3)

# 1. Drift Summary Pie Chart
ax1 = fig.add_subplot(gs[0, 0])
drifted = metric['number_of_drifted_columns']
not_drifted = metric['number_of_columns'] - drifted
colors = ['#ff6b6b', '#51cf66']
explode = (0.05, 0)
ax1.pie([drifted, not_drifted], explode=explode, labels=['Drifted', 'Stable'], 
        colors=colors, autopct='%1.1f%%', shadow=True, startangle=90)
ax1.set_title('Column Drift Status', fontsize=12, fontweight='bold')

# 2. Drift Metrics Bar Chart
ax2 = fig.add_subplot(gs[0, 1])
metrics_data = {
    'Total\nColumns': metric['number_of_columns'],
    'Drifted\nColumns': metric['number_of_drifted_columns'],
    'Stable\nColumns': metric['number_of_columns'] - metric['number_of_drifted_columns']
}
bars = ax2.bar(metrics_data.keys(), metrics_data.values(), 
               color=['#4dabf7', '#ff6b6b', '#51cf66'], edgecolor='black', linewidth=1.2)
ax2.set_ylabel('Count', fontsize=10)
ax2.set_title('Drift Metrics Overview', fontsize=12, fontweight='bold')
for bar, val in zip(bars, metrics_data.values()):
    ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5, 
             str(val), ha='center', va='bottom', fontweight='bold', fontsize=11)

# 3. Drift Share Gauge-like visualization
ax3 = fig.add_subplot(gs[0, 2])
drift_share = metric['share_of_drifted_columns']
colors_gauge = ['#51cf66' if drift_share < 0.25 else '#fcc419' if drift_share < 0.5 else '#ff6b6b']
ax3.barh(['Drift Share'], [drift_share], color=colors_gauge, height=0.4, edgecolor='black')
ax3.barh(['Drift Share'], [1-drift_share], left=[drift_share], color='#e9ecef', height=0.4, edgecolor='black')
ax3.set_xlim(0, 1)
ax3.set_ylim(-0.8, 0.8)  # Extend y-axis to make room for legend below
ax3.set_title('Drift Threshold Indicator', fontsize=12, fontweight='bold')
ax3.axvline(x=0.25, color='orange', linestyle='--', linewidth=2, label='Warning (25%)')
ax3.axvline(x=0.5, color='red', linestyle='--', linewidth=2, label='Critical (50%)')
ax3.legend(loc='lower right', fontsize=8)
ax3.text(drift_share/2, 0, f'{drift_share:.1%}', ha='center', va='center', 
         fontweight='bold', fontsize=14, color='white')

# 4+. Distribution Comparisons for columns (3 per row, starting from row 2)
for idx, col in enumerate(plot_cols):
    row = 1 + idx // 3  # Start from row 1 (0-indexed), 3 plots per row
    col_idx = idx % 3
    ax = fig.add_subplot(gs[row, col_idx])
    
    # Check if this column is drifted
    is_drifted = col in drifted_columns
    border_color = '#ff6b6b' if is_drifted else '#51cf66'
    
    # Plot distributions
    sns.kdeplot(data=reference[col], ax=ax, label='Reference', color='#339af0', 
                fill=True, alpha=0.3, linewidth=2, warn_singular=False)
    sns.kdeplot(data=current[col], ax=ax, label='Current', color='#f06595', 
                fill=True, alpha=0.3, linewidth=2, warn_singular=False)
    
    # Add drift status to title
    status = "⚠️ DRIFTED" if is_drifted else "✓ Stable"
    title_color = '#dc3545' if is_drifted else '#28a745'
    ax.set_title(f'{col[:15]}\n({status})', fontsize=9, fontweight='bold', color=title_color)
    ax.legend(fontsize=7)
    ax.set_xlabel('')
    ax.set_ylabel('Density', fontsize=8)
    
    # Add colored border for drifted columns
    for spine in ax.spines.values():
        spine.set_edgecolor(border_color)
        spine.set_linewidth(2 if is_drifted else 1)

plt.tight_layout(rect=[0, 0.02, 1, 0.96])
plt.savefig(VISUAL_REPORT, dpi=150, bbox_inches='tight', facecolor='white')
plt.close()

# -----------------------------
# Create Combined HTML Report
# -----------------------------
# Read the original Evidently HTML report
with open(HTML_REPORT, 'r', encoding='utf-8') as f:
    evidently_html = f.read()

# Encode the visualization image as base64
with open(VISUAL_REPORT, 'rb') as img_file:
    img_base64 = base64.b64encode(img_file.read()).decode('utf-8')

# Create custom header with visualization
custom_header = f'''
<div style="background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); padding: 30px; margin-bottom: 20px; border-radius: 10px; color: white; text-align: center;">
    <h1 style="margin: 0; font-size: 2.5em;">🔍 Data Drift Analysis Dashboard</h1>
    <p style="margin: 10px 0 0 0; font-size: 1.2em; opacity: 0.9;">Generated on {pd.Timestamp.now().strftime("%Y-%m-%d %H:%M:%S")}</p>
</div>

<div style="background: white; padding: 20px; margin-bottom: 30px; border-radius: 10px; box-shadow: 0 4px 6px rgba(0,0,0,0.1);">
    <h2 style="color: #333; border-bottom: 3px solid #667eea; padding-bottom: 10px;">📊 Visual Summary</h2>
    <div style="text-align: center;">
        <img src="data:image/png;base64,{img_base64}" style="max-width: 100%; height: auto; border-radius: 8px; box-shadow: 0 2px 10px rgba(0,0,0,0.1);" alt="Drift Visualizations"/>
    </div>
</div>

<div style="background: {"#ffebee" if metric["dataset_drift"] else "#e8f5e9"}; padding: 20px; margin-bottom: 30px; border-radius: 10px; border-left: 5px solid {"#f44336" if metric["dataset_drift"] else "#4caf50"};">
    <h2 style="margin: 0 0 15px 0; color: {"#c62828" if metric["dataset_drift"] else "#2e7d32"};">
        {"⚠️ Drift Detected!" if metric["dataset_drift"] else "✅ No Significant Drift"}
    </h2>
    <div style="display: flex; gap: 30px; flex-wrap: wrap;">
        <div style="flex: 1; min-width: 200px;">
            <p style="margin: 5px 0; font-size: 1.1em;"><strong>Drifted Columns:</strong> {metric["number_of_drifted_columns"]} / {metric["number_of_columns"]}</p>
            <p style="margin: 5px 0; font-size: 1.1em;"><strong>Drift Share:</strong> {metric["share_of_drifted_columns"]:.2%}</p>
        </div>
        <div style="flex: 1; min-width: 200px;">
            <p style="margin: 5px 0; font-size: 1.1em;"><strong>Severity:</strong> 
                {"🔴 HIGH - Retraining recommended!" if metric["share_of_drifted_columns"] >= 0.5 else "🟡 MEDIUM - Consider monitoring" if metric["share_of_drifted_columns"] >= 0.25 else "🟢 LOW - Acceptable"}
            </p>
        </div>
    </div>
</div>

<div style="background: white; padding: 20px; border-radius: 10px; box-shadow: 0 4px 6px rgba(0,0,0,0.1);">
    <h2 style="color: #333; border-bottom: 3px solid #667eea; padding-bottom: 10px;">📈 Detailed Evidently Report</h2>
'''

# Insert custom header after body tag
combined_html = evidently_html.replace('<body>', '<body style="background: #f5f5f5; padding: 20px;">' + custom_header)
combined_html = combined_html.replace('</body>', '</div></body>')

# Save combined report
COMBINED_REPORT = os.path.join(REPORT_DIR, "drift_report_combined.html")
with open(COMBINED_REPORT, 'w', encoding='utf-8') as f:
    f.write(combined_html)

# -----------------------------
# Console Summary with Colors
# -----------------------------
print(f"\n{'─'*50}")
drift_status = "🔴 YES" if metric['dataset_drift'] else "🟢 NO"
print(f"⚠  Dataset Drift Detected: {drift_status}")
print(f"📊 Share of Drifted Columns: {metric['share_of_drifted_columns']:.2%}")
print(f"📈 Drifted: {metric['number_of_drifted_columns']} / {metric['number_of_columns']} columns")
print(f"{'─'*50}")

# Drift severity indicator
drift_pct = metric['share_of_drifted_columns'] * 100
if drift_pct < 25:
    severity = "🟢 LOW - Acceptable drift levels"
elif drift_pct < 50:
    severity = "🟡 MEDIUM - Consider monitoring"
else:
    severity = "🔴 HIGH - Retraining recommended!"
    
print(f"📋 Severity: {severity}")
print(f"{'─'*50}")

print(f"\n✅ Reports saved:")
print(f"   📄 HTML Report: {HTML_REPORT}")
print(f"   📄 Combined Report: {COMBINED_REPORT}")
print(f"   📄 JSON Report: {JSON_REPORT}")
print(f"   📊 Visual Report: {VISUAL_REPORT}")

# Open combined HTML report in browser (only if not running headless)
if not HEADLESS:
    html_path = os.path.abspath(COMBINED_REPORT)
    webbrowser.open(f'file://{html_path}')
    print(f"\n🌐 Opening combined HTML report in browser...")
