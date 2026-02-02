import os
import json
import subprocess
import sys
import base64
import webbrowser
import shutil
from http.server import HTTPServer, SimpleHTTPRequestHandler
from urllib.parse import parse_qs
import pandas as pd
import numpy as np

DRIFT_THRESHOLD = 0.25
PORT = 8050
REPORT_DIR = "reports"

class PipelineHandler(SimpleHTTPRequestHandler):
    """Custom HTTP handler for the drift pipeline web interface"""
    
    def do_GET(self):
        if self.path == '/' or self.path == '/dashboard':
            self.serve_dashboard()
        elif self.path == '/approve':
            self.handle_approval()
        elif self.path == '/reject':
            self.handle_rejection()
        elif self.path == '/refresh':
            self.run_drift_detection()
            self.serve_dashboard()
        elif self.path == '/reset':
            self.handle_reset()
        else:
            super().do_GET()
    
    def serve_dashboard(self):
        """Serve the main dashboard with drift report and approval buttons"""
        html = self.generate_dashboard_html()
        self.send_response(200)
        self.send_header('Content-type', 'text/html')
        self.end_headers()
        self.wfile.write(html.encode('utf-8'))
    
    def handle_approval(self):
        """Handle retraining approval"""
        print("\n✅ Retraining approved by user via web interface!")
        
        # Run retraining
        try:
            subprocess.run([sys.executable, "src/retrain_model.py"], check=True)
            message = "✅ Model retrained successfully!"
            success = True
        except subprocess.CalledProcessError as e:
            message = f"❌ Retraining failed: {e}"
            success = False
        
        # Re-run drift detection to update visuals with ALL columns shown
        if success:
            print("🔄 Regenerating drift visualizations with all columns...")
            try:
                subprocess.run([sys.executable, "src/detect_drift.py", "--no-browser", "--show-all"], check=True)
                print("✅ Visualizations updated with all columns!")
            except subprocess.CalledProcessError as e:
                print(f"⚠️ Warning: Could not regenerate visuals: {e}")
        
        # Serve updated dashboard with message
        html = self.generate_dashboard_html(message=message, success=success)
        self.send_response(200)
        self.send_header('Content-type', 'text/html')
        self.end_headers()
        self.wfile.write(html.encode('utf-8'))
    
    def handle_rejection(self):
        """Handle retraining rejection"""
        print("\n✋ Retraining rejected by user via web interface")
        message = "✋ Retraining rejected. No changes made."
        html = self.generate_dashboard_html(message=message, success=False)
        self.send_response(200)
        self.send_header('Content-type', 'text/html')
        self.end_headers()
        self.wfile.write(html.encode('utf-8'))
    
    def run_drift_detection(self):
        """Run drift detection script"""
        try:
            subprocess.run([sys.executable, "src/detect_drift.py", "--no-browser"], check=True,
                         capture_output=True)
        except subprocess.CalledProcessError as e:
            print(f"Error running drift detection: {e}")
    
    def handle_reset(self):
        """Reset reference data to simulate high drift for demo"""
        print("\n🔄 Resetting to demo state (high drift)...")
        
        try:
            # Create synthetic reference data with different distribution
            current_df = pd.read_csv("data/current.csv")
            
            # Create reference with shifted distributions to cause drift
            reference_df = current_df.copy()
            numeric_cols = reference_df.select_dtypes(include=['number']).columns
            
            for col in numeric_cols:
                if col != 'Churn':  # Don't modify target
                    # Shift the distribution to create drift
                    reference_df[col] = reference_df[col] * np.random.uniform(0.5, 1.5) + np.random.uniform(-10, 10)
            
            # Take only first 500 rows to make it different
            reference_df = reference_df.iloc[:500]
            reference_df.to_csv("data/reference.csv", index=False)
            
            # Re-run drift detection
            subprocess.run([sys.executable, "src/detect_drift.py", "--no-browser"], check=True)
            
            message = "🔄 Reset complete! High drift state restored. Review and approve retraining."
            success = True
            print("✅ Reset complete - high drift state restored")
        except Exception as e:
            message = f"❌ Reset failed: {e}"
            success = False
            print(f"❌ Reset failed: {e}")
        
        html = self.generate_dashboard_html(message=message, success=success, is_reset=True)
        self.send_response(200)
        self.send_header('Content-type', 'text/html')
        self.end_headers()
        self.wfile.write(html.encode('utf-8'))
    
    def generate_dashboard_html(self, message=None, success=None, is_reset=False):
        """Generate the dashboard HTML with embedded visualizations"""
        
        # Load drift report data
        try:
            with open("reports/drift_report.json") as f:
                report = json.load(f)
            result = report["metrics"][0]["result"]
            drift_ratio = result["number_of_drifted_columns"] / result["number_of_columns"]
            has_drift_data = True
        except:
            result = {}
            drift_ratio = 0
            has_drift_data = False
        
        # Load visualization image as base64
        img_base64 = ""
        try:
            with open("reports/drift_visualizations.png", "rb") as img_file:
                img_base64 = base64.b64encode(img_file.read()).decode('utf-8')
        except:
            pass
        
        # Determine severity
        if drift_ratio >= 0.5:
            severity = "🔴 HIGH"
            severity_color = "#dc3545"
            severity_bg = "#f8d7da"
        elif drift_ratio >= 0.25:
            severity = "🟡 MEDIUM"
            severity_color = "#ffc107"
            severity_bg = "#fff3cd"
        else:
            severity = "🟢 LOW"
            severity_color = "#28a745"
            severity_bg = "#d4edda"
        
        needs_action = drift_ratio >= DRIFT_THRESHOLD
        
        # Message banner
        message_html = ""
        if message:
            if is_reset:
                msg_color = "#17a2b8"  # Info blue for reset
                msg_bg = "#d1ecf1"
            else:
                msg_color = "#28a745" if success else "#dc3545"
                msg_bg = "#d4edda" if success else "#f8d7da"
            message_html = f'''
            <div style="background: {msg_bg}; color: {msg_color}; padding: 20px; 
                        border-radius: 10px; margin-bottom: 20px; text-align: center;
                        font-size: 1.3em; font-weight: bold; border: 2px solid {msg_color};">
                {message}
            </div>
            '''
        
        html = f'''
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>🔍 Drift Detection Pipeline</title>
    <style>
        * {{ margin: 0; padding: 0; box-sizing: border-box; }}
        body {{
            font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
            background: linear-gradient(135deg, #1a1a2e 0%, #16213e 100%);
            min-height: 100vh;
            padding: 20px;
            color: #333;
        }}
        .container {{
            max-width: 1400px;
            margin: 0 auto;
        }}
        .header {{
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            padding: 30px;
            border-radius: 15px;
            color: white;
            text-align: center;
            margin-bottom: 20px;
            box-shadow: 0 10px 30px rgba(102, 126, 234, 0.4);
        }}
        .header h1 {{
            font-size: 2.5em;
            margin-bottom: 10px;
        }}
        .header p {{
            opacity: 0.9;
            font-size: 1.1em;
        }}
        .card {{
            background: white;
            border-radius: 15px;
            padding: 25px;
            margin-bottom: 20px;
            box-shadow: 0 5px 20px rgba(0,0,0,0.1);
        }}
        .card h2 {{
            color: #333;
            border-bottom: 3px solid #667eea;
            padding-bottom: 10px;
            margin-bottom: 20px;
        }}
        .metrics-grid {{
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
            gap: 20px;
            margin-bottom: 20px;
        }}
        .metric-box {{
            background: linear-gradient(135deg, #f8f9fa 0%, #e9ecef 100%);
            padding: 20px;
            border-radius: 10px;
            text-align: center;
            border-left: 4px solid #667eea;
        }}
        .metric-box .value {{
            font-size: 2.5em;
            font-weight: bold;
            color: #333;
        }}
        .metric-box .label {{
            color: #666;
            font-size: 0.9em;
            margin-top: 5px;
        }}
        .severity-badge {{
            display: inline-block;
            padding: 10px 20px;
            border-radius: 25px;
            font-weight: bold;
            font-size: 1.2em;
            background: {severity_bg};
            color: {severity_color};
            border: 2px solid {severity_color};
        }}
        .action-section {{
            background: {"#fff3cd" if needs_action else "#d4edda"};
            border: 2px solid {"#ffc107" if needs_action else "#28a745"};
            border-radius: 15px;
            padding: 30px;
            text-align: center;
            margin-bottom: 20px;
        }}
        .action-section h3 {{
            color: {"#856404" if needs_action else "#155724"};
            font-size: 1.5em;
            margin-bottom: 20px;
        }}
        .btn {{
            display: inline-block;
            padding: 15px 40px;
            font-size: 1.2em;
            font-weight: bold;
            border: none;
            border-radius: 50px;
            cursor: pointer;
            text-decoration: none;
            margin: 10px;
            transition: all 0.3s ease;
            box-shadow: 0 4px 15px rgba(0,0,0,0.2);
        }}
        .btn-approve {{
            background: linear-gradient(135deg, #28a745 0%, #20c997 100%);
            color: white;
        }}
        .btn-approve:hover {{
            transform: translateY(-3px);
            box-shadow: 0 6px 20px rgba(40, 167, 69, 0.4);
        }}
        .btn-reject {{
            background: linear-gradient(135deg, #dc3545 0%, #c82333 100%);
            color: white;
        }}
        .btn-reject:hover {{
            transform: translateY(-3px);
            box-shadow: 0 6px 20px rgba(220, 53, 69, 0.4);
        }}
        .btn-refresh {{
            background: linear-gradient(135deg, #17a2b8 0%, #138496 100%);
            color: white;
        }}
        .btn-refresh:hover {{
            transform: translateY(-3px);
            box-shadow: 0 6px 20px rgba(23, 162, 184, 0.4);
        }}
        .btn-reset {{
            background: linear-gradient(135deg, #6c757d 0%, #495057 100%);
            color: white;
        }}
        .btn-reset:hover {{
            transform: translateY(-3px);
            box-shadow: 0 6px 20px rgba(108, 117, 125, 0.4);
        }}
        .visualization {{
            text-align: center;
        }}
        .visualization img {{
            max-width: 100%;
            border-radius: 10px;
            box-shadow: 0 5px 20px rgba(0,0,0,0.1);
        }}
        .timestamp {{
            text-align: center;
            color: #666;
            font-size: 0.9em;
            margin-top: 20px;
        }}
        .no-action {{
            background: #d4edda;
            border: 2px solid #28a745;
        }}
        .no-action h3 {{
            color: #155724;
        }}
    </style>
</head>
<body>
    <div class="container">
        <div class="header">
            <h1>🔍 Human-in-the-Loop Drift Pipeline</h1>
            <p>Monitor data drift and approve model retraining</p>
        </div>
        
        {message_html}
        
        <div class="card">
            <h2>📊 Drift Metrics</h2>
            <div class="metrics-grid">
                <div class="metric-box">
                    <div class="value">{result.get("number_of_columns", "N/A")}</div>
                    <div class="label">Total Columns</div>
                </div>
                <div class="metric-box">
                    <div class="value" style="color: #dc3545;">{result.get("number_of_drifted_columns", "N/A")}</div>
                    <div class="label">Drifted Columns</div>
                </div>
                <div class="metric-box">
                    <div class="value">{drift_ratio:.1%}</div>
                    <div class="label">Drift Ratio</div>
                </div>
                <div class="metric-box">
                    <div class="value">{DRIFT_THRESHOLD:.0%}</div>
                    <div class="label">Threshold</div>
                </div>
            </div>
            <div style="text-align: center; margin-top: 20px;">
                <span class="severity-badge">{severity} Severity</span>
            </div>
        </div>
        
        <div class="action-section {"" if needs_action else "no-action"}">
            {"<h3>⚠️ Drift exceeds threshold! Human approval required for retraining.</h3>" if needs_action else "<h3>✅ Drift is within acceptable limits. No action required.</h3>"}
            <div>
                {"<a href='/approve' class='btn btn-approve'>✅ Approve Retraining</a><a href='/reject' class='btn btn-reject'>❌ Reject</a>" if needs_action else ""}
                <a href="/refresh" class="btn btn-refresh">🔄 Refresh Analysis</a>
                <a href="/reset" class="btn btn-reset">⚡ Reset Demo</a>
            </div>
        </div>
        
        <div class="card">
            <h2>📈 Visual Analysis</h2>
            <div class="visualization">
                {"<img src='data:image/png;base64," + img_base64 + "' alt='Drift Visualizations'/>" if img_base64 else "<p>No visualization available. Run drift detection first.</p>"}
            </div>
        </div>
        
        <div class="timestamp">
            Last updated: {pd.Timestamp.now().strftime("%Y-%m-%d %H:%M:%S")}
        </div>
    </div>
</body>
</html>
'''
        return html
    
    def log_message(self, format, *args):
        """Suppress default logging"""
        pass


def run_pipeline():
    """Main pipeline function"""
    print("\n" + "="*60)
    print("🚀 Starting Human-in-the-Loop Drift Pipeline (Web Interface)")
    print("="*60)
    
    # Run initial drift detection (with --no-browser to avoid opening HTML file)
    print("\n📊 Running drift detection...")
    try:
        subprocess.run([sys.executable, "src/detect_drift.py", "--no-browser"], check=True)
    except subprocess.CalledProcessError as e:
        print(f"❌ Error running drift detection: {e}")
        return
    
    # Load drift report
    try:
        with open("reports/drift_report.json") as f:
            report = json.load(f)
        result = report["metrics"][0]["result"]
        drift_ratio = result["number_of_drifted_columns"] / result["number_of_columns"]
    except Exception as e:
        print(f"❌ Error loading drift report: {e}")
        return
    
    print(f"\n📊 Drift ratio: {drift_ratio:.2%}")
    print(f"📊 Threshold: {DRIFT_THRESHOLD:.2%}")
    
    if drift_ratio >= DRIFT_THRESHOLD:
        print(f"\n⚠️ Drift exceeds threshold! Starting web interface for human approval...")
    else:
        print(f"\n✅ Drift is acceptable. Starting web interface for monitoring...")
    
    # Start web server
    server = HTTPServer(('localhost', PORT), PipelineHandler)
    
    print(f"\n🌐 Web interface available at: http://localhost:{PORT}")
    print("📝 Open the URL in your browser to review and approve/reject retraining")
    print("🛑 Press Ctrl+C to stop the server\n")
    
    # Open browser automatically
    webbrowser.open(f'http://localhost:{PORT}')
    
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\n\n🛑 Server stopped by user")
        server.shutdown()


if __name__ == "__main__":
    run_pipeline()
