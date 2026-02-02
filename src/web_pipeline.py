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

# Import CLI pipeline functions to avoid code duplication
from pipeline import approve_retraining, reset_demo, run_drift_detection, get_drift_status, auto_resolve_drift, DRIFT_THRESHOLD

PORT = 8050
REPORT_DIR = "reports"

class PipelineHandler(SimpleHTTPRequestHandler):
    """Custom HTTP handler for the drift pipeline web interface"""
    
    def do_GET(self):
        if self.path == '/' or self.path.startswith('/dashboard'):
            self.serve_dashboard()
        elif self.path == '/approve':
            self.handle_approval()
        elif self.path == '/reject':
            self.handle_rejection()
        elif self.path == '/refresh':
            self.do_run_drift_detection()
            self.serve_dashboard()
        elif self.path == '/reset':
            self.handle_reset()
        elif self.path == '/auto_resolve':
            self.handle_auto_resolve()
        else:
            super().do_GET()
    
    def serve_dashboard(self):
        """Serve the main dashboard with drift report and approval buttons"""
        # Check for message and gif flag in query string
        message = None
        success = True
        show_gif = False
        
        if '?' in self.path:
            from urllib.parse import unquote, parse_qs, urlparse
            parsed = urlparse(self.path)
            params = parse_qs(parsed.query)
            
            if 'msg' in params:
                message = unquote(params['msg'][0].replace('+', ' '))
                success = '✅' in message
            
            if 'gif' in params:
                show_gif = True
        
        html = self.generate_dashboard_html(message=message, success=success, show_gif=show_gif)
        self.send_response(200)
        self.send_header('Content-type', 'text/html')
        self.send_header('Cache-Control', 'no-cache, no-store, must-revalidate')
        self.end_headers()
        self.wfile.write(html.encode('utf-8'))
    
    def handle_approval(self):
        """Handle retraining approval by calling CLI function"""
        print("\n✅ Retraining approved by user via web interface!")
        
        # Call the CLI pipeline function (reuses logic, no duplication)
        result = approve_retraining(silent=True)
        
        # Redirect to dashboard to force browser refresh with new data
        from urllib.parse import quote
        msg = quote(result["message"])
        self.send_response(302)
        self.send_header('Location', f'/dashboard?msg={msg}')
        self.end_headers()
    
    def handle_rejection(self):
        """Handle retraining rejection"""
        print("\n✋ Retraining rejected by user via web interface")
        message = "✋ Retraining rejected. No changes made."
        html = self.generate_dashboard_html(message=message, success=False)
        self.send_response(200)
        self.send_header('Content-type', 'text/html')
        self.end_headers()
        self.wfile.write(html.encode('utf-8'))
    
    def do_run_drift_detection(self):
        """Run drift detection script using CLI function"""
        run_drift_detection(show_all=True, open_browser=False)
    
    def handle_reset(self):
        """Reset reference data to simulate high drift for demo"""
        # Use CLI function (no duplication)
        result = reset_demo(silent=True)
        
        html = self.generate_dashboard_html(message=result["message"], success=result["success"], is_reset=True)
        self.send_response(200)
        self.send_header('Content-type', 'text/html')
        self.end_headers()
        self.wfile.write(html.encode('utf-8'))
    
    def handle_auto_resolve(self):
        """Handle automatic drift resolution until drift is acceptable"""
        print("\n🤖 Auto-resolving drift via web interface...")
        
        # Call the CLI auto_resolve function (reuses logic, no duplication)
        result = auto_resolve_drift(silent=True)
        
        # Redirect to dashboard to force browser refresh with new data
        # Include gif flag if GIF was created
        from urllib.parse import quote
        msg = quote(result["message"])
        gif_param = "&gif=1" if result.get("gif_path") else ""
        self.send_response(302)
        self.send_header('Location', f'/dashboard?msg={msg}{gif_param}')
        self.end_headers()
    
    def generate_dashboard_html(self, message=None, success=None, is_reset=False, show_gif=False):
        """Generate the dashboard HTML from template with dynamic data"""
        
        # Load HTML template
        try:
            with open("src/templates/dashboard.html", "r", encoding="utf-8") as f:
                html_template = f.read()
        except FileNotFoundError:
            return "<html><body><h1>Error: Template file not found</h1></body></html>"
        
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
        
        # Load progress GIF as base64 if requested
        gif_base64 = ""
        if show_gif:
            try:
                with open("reports/auto_resolve_progress.gif", "rb") as gif_file:
                    gif_base64 = base64.b64encode(gif_file.read()).decode('utf-8')
            except:
                pass
        
        # Determine severity
        if drift_ratio >= 0.25:
            severity = "🔴 HIGH"
            severity_color = "#dc3545"
            severity_bg = "#f8d7da"
        elif drift_ratio >= 0.10:
            severity = "🟡 MEDIUM"
            severity_color = "#ffc107"
            severity_bg = "#fff3cd"
        else:
            severity = "🟢 LOW"
            severity_color = "#28a745"
            severity_bg = "#d4edda"
        
        needs_action = drift_ratio >= DRIFT_THRESHOLD
        
        # Action section colors
        action_bg = "#fff3cd" if needs_action else "#d4edda"
        action_border = "#ffc107" if needs_action else "#28a745"
        action_text_color = "#856404" if needs_action else "#155724"
        
        # Action message and buttons (web interface has all interactive buttons)
        if needs_action:
            action_message = "⚠️ Drift exceeds threshold! Human approval required for retraining."
            action_buttons = """<a href='/approve' class='btn btn-approve'>✅ Approve Retraining</a>
                <a href='/auto_resolve' class='btn btn-auto'>🤖 Auto Resolve</a>
                <a href="/refresh" class="btn btn-refresh">🔄 Refresh Analysis</a>
                <a href="/reset" class="btn btn-reset">⚡ Reset Demo</a>"""
        else:
            action_message = "✅ Drift is within acceptable limits. No action required."
            action_buttons = """<a href="/refresh" class="btn btn-refresh">🔄 Refresh Analysis</a>
                <a href="/reset" class="btn btn-reset">⚡ Reset Demo</a>"""
        
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
        
        # Visualization HTML - show GIF if available (after auto-resolve), otherwise show PNG
        if gif_base64:
            visualization_html = f"<img src='data:image/gif;base64,{gif_base64}' alt='Auto-Resolve Progress Timeline'/>"
        elif img_base64:
            visualization_html = f"<img src='data:image/png;base64,{img_base64}' alt='Drift Visualizations'/>"
        else:
            visualization_html = "<p>No visualization available. Run drift detection first.</p>"
        
        # Replace placeholders in template
        html = html_template.replace("{{MESSAGE_HTML}}", message_html)
        html = html.replace("{{TOTAL_COLUMNS}}", str(result.get("number_of_columns", "N/A")))
        html = html.replace("{{DRIFTED_COLUMNS}}", str(result.get("number_of_drifted_columns", "N/A")))
        html = html.replace("{{DRIFT_RATIO}}", f"{drift_ratio:.1%}")
        html = html.replace("{{THRESHOLD}}", f"{DRIFT_THRESHOLD:.0%}")
        html = html.replace("{{SEVERITY}}", severity)
        html = html.replace("{{SEVERITY_COLOR}}", severity_color)
        html = html.replace("{{SEVERITY_BG}}", severity_bg)
        html = html.replace("{{ACTION_BG}}", action_bg)
        html = html.replace("{{ACTION_BORDER}}", action_border)
        html = html.replace("{{ACTION_TEXT_COLOR}}", action_text_color)
        html = html.replace("{{ACTION_MESSAGE}}", action_message)
        html = html.replace("{{ACTION_BUTTONS}}", action_buttons)
        html = html.replace("{{VISUALIZATION}}", visualization_html)
        html = html.replace("{{TIMESTAMP}}", pd.Timestamp.now().strftime("%Y-%m-%d %H:%M:%S"))
        
        return html
    
    def log_message(self, format, *args):
        """Suppress default logging"""
        pass


def run_pipeline():
    """Main pipeline function"""
    print("\n" + "="*60)
    print("🚀 Starting Human-in-the-Loop Drift Pipeline (Web Interface)")
    print("="*60)
    
    # Always reset to high drift state on startup for demo
    print("\n🔄 Initializing demo with high drift state...")
    reset_demo(silent=True)
    
    # Load drift report using CLI function
    drift_status = get_drift_status()
    if not drift_status:
        print("❌ Error loading drift report")
        return
    
    drift_ratio = drift_status["drift_ratio"]
    
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
