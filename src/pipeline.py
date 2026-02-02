import json
import subprocess
import sys
import random
import os
import shutil
from datetime import datetime
import pandas as pd
import numpy as np
from PIL import Image

DRIFT_THRESHOLD = 0.10

def run_drift_detection(show_all=True, open_browser=False):
    """Run drift detection script
    
    Args:
        show_all (bool): Show all columns in visualization
        open_browser (bool): Open browser with report
    
    Returns:
        bool: True if successful, False otherwise
    """
    try:
        args = [sys.executable, "src/detect_drift.py"]
        if not open_browser:
            args.append("--no-browser")
        if show_all:
            args.append("--show-all")
        subprocess.run(args, check=True)
        return True
    except subprocess.CalledProcessError as e:
        print(f"Error running drift detection: {e}")
        return False

def reset_demo(silent=False):
    """Reset reference data to simulate high drift for demo
    
    Args:
        silent (bool): If True, suppress print statements
    
    Returns:
        dict: Result with success status and message
    """
    if not silent:
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
        run_drift_detection(show_all=True, open_browser=False)
        
        message = "🔄 Reset complete! High drift state restored. Review and approve retraining."
        if not silent:
            print("✅ Reset complete - high drift state restored")
        
        return {"success": True, "message": message}
    except Exception as e:
        message = f"❌ Reset failed: {e}"
        if not silent:
            print(message)
        return {"success": False, "message": message}

def get_drift_status():
    """Load and return current drift status"""
    try:
        with open("reports/drift_report.json") as f:
            report = json.load(f)
        
        result = report["metrics"][0]["result"]
        num_drifted = result["number_of_drifted_columns"]
        total_cols = result["number_of_columns"]
        drift_ratio = num_drifted / total_cols if total_cols > 0 else 0
        
        return {
            "num_drifted": num_drifted,
            "total_cols": total_cols,
            "drift_ratio": drift_ratio,
            "report": report
        }
    except Exception as e:
        print(f"⚠️  Error loading drift status: {e}")
        return None

def approve_retraining(silent=False):
    """Handle retraining approval with gradual drift reduction
    
    Args:
        silent (bool): If True, suppress print statements for web interface
    
    Returns:
        dict: Result with status, message, and updated metrics
    """
    # Load current drift state
    drift_status = get_drift_status()
    if not drift_status:
        return {
            "success": False,
            "message": "❌ Could not load drift status",
            "num_drifted": 0,
            "drift_ratio": 0
        }
    
    num_drifted = drift_status["num_drifted"]
    total_cols = drift_status["total_cols"]
    
    # FIRST: Simulate gradual drift reduction by replacing reference rows with current rows
    # This creates ACTUAL reduced drift that will show in visualizations
    reduction_factor = random.uniform(0.4, 0.6)  # 40-60% of rows replaced per approval
    
    try:
        reference_df = pd.read_csv("data/reference.csv")
        current_df = pd.read_csv("data/current.csv")
        
        # Calculate how many rows to replace
        num_rows_to_replace = int(len(reference_df) * reduction_factor)
        
        # Get random indices to replace
        replace_indices = random.sample(range(len(reference_df)), num_rows_to_replace)
        
        # Sample rows from current to use as replacements
        replacement_rows = current_df.sample(n=num_rows_to_replace, replace=True).reset_index(drop=True)
        
        # Replace rows in reference with rows from current
        for i, idx in enumerate(replace_indices):
            reference_df.iloc[idx] = replacement_rows.iloc[i]
        
        # Save updated reference BEFORE retraining
        reference_df.to_csv("data/reference.csv", index=False)
        
        if not silent:
            print(f"📊 Replaced {num_rows_to_replace} rows ({reduction_factor:.0%}) of reference with current data")
    except Exception as e:
        if not silent:
            print(f"⚠️  Warning: Could not update reference data: {e}")
    
    # THEN: Run retraining on the blended data
    try:
        subprocess.run([sys.executable, "src/retrain_model.py"], check=True)
    except subprocess.CalledProcessError as e:
        message = f"❌ Retraining failed: {e}"
        if not silent:
            print(message)
        return {
            "success": False,
            "message": message,
            "num_drifted": num_drifted,
            "drift_ratio": drift_status["drift_ratio"] * 100
        }
    
    # Re-run drift detection to generate new visualization with reduced drift
    try:
        # Always run without capture_output to ensure it completes
        subprocess.run([sys.executable, "src/detect_drift.py", "--no-browser", "--show-all"], 
                      check=True)
        if not silent:
            print("📊 New drift visualization generated")
    except subprocess.CalledProcessError as e:
        if not silent:
            print(f"⚠️  Warning: Could not regenerate visualization: {e}")
    
    # Get updated drift status after regeneration
    new_status = get_drift_status()
    if new_status:
        new_drifted = new_status["num_drifted"]
        new_drift_ratio = new_status["drift_ratio"] * 100
    else:
        new_drifted = int(num_drifted * (1 - reduction_factor))
        new_drift_ratio = (new_drifted / total_cols) * 100 if total_cols > 0 else 0
    
    # Add retrain info overlay to the visualization
    add_retrain_overlay_to_visualization(new_drift_ratio / 100, new_drifted, total_cols)
    
    # Check if drift is fully resolved
    if new_drifted <= 1 or new_drift_ratio < 5.0:
        message = f"✅ Model retrained successfully! Drift fully resolved (reduced from {num_drifted} to {new_drifted} columns)"
        resolved = True
    else:
        message = f"⚠️  Model retrained. Drift reduced from {num_drifted} to {new_drifted} columns ({new_drift_ratio:.1f}%). Continue monitoring..."
        resolved = False
    
    if not silent:
        print(f"\n{message}")
    
    return {
        "success": True,
        "message": message,
        "num_drifted": new_drifted,
        "drift_ratio": new_drift_ratio,
        "resolved": resolved
    }

def add_retrain_overlay_to_visualization(drift_ratio, num_drifted, total_cols, retrain_text="Retrained"):
    """Add retrain info overlay to the drift visualization PNG
    
    Args:
        drift_ratio: Current drift ratio (0-1)
        num_drifted: Number of drifted columns
        total_cols: Total number of columns
        retrain_text: Text to show (e.g., "Retrained", "Retrain #1")
    """
    from PIL import Image, ImageDraw, ImageFont
    
    viz_path = "reports/drift_visualizations.png"
    try:
        img = Image.open(viz_path).convert('RGB')
    except FileNotFoundError:
        return  # No visualization to overlay
    
    draw = ImageDraw.Draw(img)
    width, height = img.size
    
    # Try to use a nice font, fall back to default
    try:
        large_font = ImageFont.truetype("arial.ttf", 32)
    except:
        large_font = ImageFont.load_default()
    
    # Drift info
    drift_pct = drift_ratio * 100
    if drift_ratio >= 0.25:
        drift_color = '#dc3545'  # Red
    elif drift_ratio >= 0.10:
        drift_color = '#ffc107'  # Yellow  
    else:
        drift_color = '#28a745'  # Green
    
    info_text = f"{retrain_text}  |  Drift: {drift_pct:.1f}% ({num_drifted}/{total_cols} columns)"
    
    # Position text centered, below the title
    text_y = 85
    
    # Draw background rectangle for better visibility
    bbox = draw.textbbox((width // 2, text_y), info_text, font=large_font, anchor='mm')
    padding = 10
    draw.rectangle(
        [bbox[0] - padding, bbox[1] - padding, bbox[2] + padding, bbox[3] + padding],
        fill='white',
        outline=drift_color,
        width=3
    )
    
    # Draw the text
    draw.text((width // 2, text_y), info_text, fill=drift_color, font=large_font, anchor='mm')
    
    img.save(viz_path)

def create_progress_frame(iteration, drift_ratio, num_drifted, total_cols, status_text, frame_path):
    """Create a progress frame by copying the current drift visualization with retrain number under title
    
    Args:
        iteration: Current iteration number (0 = initial state)
        drift_ratio: Current drift ratio (0-1)
        num_drifted: Number of drifted columns
        total_cols: Total number of columns
        status_text: Status message to display
        frame_path: Path to save the frame image
    """
    from PIL import Image, ImageDraw, ImageFont
    
    # Load the current drift visualization
    viz_path = "reports/drift_visualizations.png"
    try:
        img = Image.open(viz_path).convert('RGB')
    except FileNotFoundError:
        # Create a placeholder if visualization doesn't exist
        img = Image.new('RGB', (1500, 1200), color='#f8f9fa')
    
    draw = ImageDraw.Draw(img)
    width, height = img.size
    
    # Try to use a nice font, fall back to default
    try:
        large_font = ImageFont.truetype("arial.ttf", 32)
        medium_font = ImageFont.truetype("arial.ttf", 24)
    except:
        large_font = ImageFont.load_default()
        medium_font = large_font
    
    # Retrain number text
    if iteration == 0:
        retrain_text = "Initial State"
    else:
        retrain_text = f"Retrain #{iteration}"
    
    # Drift info
    drift_pct = drift_ratio * 100
    if drift_ratio >= 0.25:
        drift_color = '#dc3545'  # Red
    elif drift_ratio >= 0.10:
        drift_color = '#ffc107'  # Yellow  
    else:
        drift_color = '#28a745'  # Green
    
    info_text = f"{retrain_text}  |  Drift: {drift_pct:.1f}% ({num_drifted}/{total_cols} columns)"
    
    # Add text right under the main title ("Data Drift Analysis Dashboard")
    # The title is at y=0.98 of figure, which is near the top. Add text below it.
    # Position text centered, below the title
    text_y = 85
    
    # Draw background rectangle for better visibility
    bbox = draw.textbbox((width // 2, text_y), info_text, font=large_font, anchor='mm')
    padding = 10
    draw.rectangle(
        [bbox[0] - padding, bbox[1] - padding, bbox[2] + padding, bbox[3] + padding],
        fill='white',
        outline=drift_color,
        width=3
    )
    
    # Draw the text
    draw.text((width // 2, text_y), info_text, fill=drift_color, font=large_font, anchor='mm')
    
    img.save(frame_path)
    return frame_path

def create_progress_gif(frame_paths, output_path, duration_per_frame=2000):
    """Create a GIF from progress frames
    
    Args:
        frame_paths: List of paths to frame images
        output_path: Path to save the GIF
        duration_per_frame: Duration per frame in milliseconds (default 2 seconds)
    """
    if not frame_paths:
        return None
    
    frames = [Image.open(fp) for fp in frame_paths]
    
    # Save as GIF with slow animation
    frames[0].save(
        output_path,
        save_all=True,
        append_images=frames[1:],
        duration=duration_per_frame,  # 2 seconds per frame
        loop=0  # Loop forever
    )
    
    # Clean up individual frames
    for fp in frame_paths:
        try:
            os.remove(fp)
        except:
            pass
    
    return output_path

def auto_resolve_drift(silent=False, max_iterations=10):
    """Automatically approve retraining until drift is acceptable
    
    Args:
        silent (bool): If True, suppress print statements
        max_iterations (int): Maximum number of retraining attempts (default 10)
    
    Returns:
        dict: Result with final status and number of iterations
    """
    if not silent:
        print("\n🤖 Starting automatic drift resolution...")
        print("📹 Recording progress timeline...")
    
    # Create frames directory
    frames_dir = "reports/gif_frames"
    os.makedirs(frames_dir, exist_ok=True)
    frame_paths = []
    
    # Run drift detection to generate initial visualization
    run_drift_detection(show_all=True, open_browser=False)
    
    # Get initial drift status for first frame
    drift_status = get_drift_status()
    if not drift_status:
        return {"success": False, "message": "❌ Could not load drift status", "iterations": 0}
    
    total_cols = drift_status["total_cols"]
    
    # Capture initial state (iteration 0) - uses current drift_visualizations.png
    initial_frame = os.path.join(frames_dir, "frame_000.png")
    create_progress_frame(
        iteration=0,
        drift_ratio=drift_status["drift_ratio"],
        num_drifted=drift_status["num_drifted"],
        total_cols=total_cols,
        status_text="⏳ Starting auto-resolution...",
        frame_path=initial_frame
    )
    frame_paths.append(initial_frame)
    
    for iteration in range(1, max_iterations + 1):
        # Get current drift status
        drift_status = get_drift_status()
        if not drift_status:
            return {"success": False, "message": "❌ Could not load drift status", "iterations": iteration}
        
        drift_ratio = drift_status["drift_ratio"]
        num_drifted = drift_status["num_drifted"]
        
        if not silent:
            print(f"\n📊 Iteration {iteration}: Drift at {drift_ratio:.1%} ({num_drifted} columns)")
        
        # Check if drift is already acceptable
        if drift_ratio < DRIFT_THRESHOLD:
            # Capture final success frame
            final_frame = os.path.join(frames_dir, f"frame_{iteration:03d}.png")
            create_progress_frame(
                iteration=iteration,
                drift_ratio=drift_ratio,
                num_drifted=num_drifted,
                total_cols=total_cols,
                status_text="✅ Drift resolved! Model aligned.",
                frame_path=final_frame
            )
            frame_paths.append(final_frame)
            
            # Create the GIF
            gif_path = "reports/auto_resolve_progress.gif"
            create_progress_gif(frame_paths, gif_path, duration_per_frame=2000)
            
            message = f"✅ Drift resolved after {iteration - 1} retraining(s)! Final drift: {drift_ratio:.1%}"
            if not silent:
                print(f"\n{message}")
                print(f"📹 Progress GIF saved to: {gif_path}")
            
            return {
                "success": True,
                "message": message,
                "iterations": iteration - 1,
                "final_drift_ratio": drift_ratio * 100,
                "final_drifted_columns": num_drifted,
                "gif_path": gif_path
            }
        
        # Approve retraining
        if not silent:
            print(f"🔧 Auto-approving retraining #{iteration}...")
        
        result = approve_retraining(silent=True)
        
        if not result["success"]:
            return {"success": False, "message": result["message"], "iterations": iteration}
        
        # Check if drift is now below threshold after this retraining
        current_drift_ratio = result["drift_ratio"] / 100  # Convert back to 0-1
        
        # Capture frame after retraining
        frame_path = os.path.join(frames_dir, f"frame_{iteration:03d}.png")
        create_progress_frame(
            iteration=iteration,
            drift_ratio=current_drift_ratio,
            num_drifted=result["num_drifted"],
            total_cols=total_cols,
            status_text=f"🔧 Retraining #{iteration} complete",
            frame_path=frame_path
        )
        frame_paths.append(frame_path)
        
        if not silent:
            print(f"   → Drift reduced to {result['drift_ratio']:.1f}% ({result['num_drifted']} columns)")
        
        # Stop immediately if drift is below threshold (10%)
        if current_drift_ratio < DRIFT_THRESHOLD:
            # Create the GIF
            gif_path = "reports/auto_resolve_progress.gif"
            create_progress_gif(frame_paths, gif_path, duration_per_frame=2000)
            
            message = f"✅ Drift resolved after {iteration} retraining(s)! Final drift: {current_drift_ratio:.1%}"
            if not silent:
                print(f"\n{message}")
                print(f"📹 Progress GIF saved to: {gif_path}")
            
            return {
                "success": True,
                "message": message,
                "iterations": iteration,
                "final_drift_ratio": result["drift_ratio"],
                "final_drifted_columns": result["num_drifted"],
                "gif_path": gif_path
            }
    
    # Max iterations reached - capture final frame
    final_status = get_drift_status()
    final_ratio = final_status["drift_ratio"] if final_status else 0
    
    final_frame = os.path.join(frames_dir, f"frame_final.png")
    create_progress_frame(
        iteration=max_iterations,
        drift_ratio=final_ratio,
        num_drifted=final_status["num_drifted"] if final_status else 0,
        total_cols=total_cols,
        status_text="⚠️ Max iterations reached",
        frame_path=final_frame
    )
    frame_paths.append(final_frame)
    
    # Create the GIF
    gif_path = "reports/auto_resolve_progress.gif"
    create_progress_gif(frame_paths, gif_path, duration_per_frame=2000)
    
    message = f"⚠️ Max iterations ({max_iterations}) reached. Final drift: {final_ratio:.1%}"
    if not silent:
        print(f"\n{message}")
        print(f"📹 Progress GIF saved to: {gif_path}")
    
    return {
        "success": final_ratio < DRIFT_THRESHOLD,
        "message": message,
        "iterations": max_iterations,
        "final_drift_ratio": final_ratio * 100,
        "final_drifted_columns": final_status["num_drifted"] if final_status else 0,
        "gif_path": gif_path
    }

def run_pipeline():
    """Main CLI pipeline with gradual drift reduction"""
    
    while True:  # Loop to allow reset and re-check
        print("\n🚀 Starting Human-in-the-Loop Drift Pipeline")
        
        # Run drift detection (always run to show current state with all columns)
        run_drift_detection(show_all=True, open_browser=False)
        
        # Get drift status
        drift_status = get_drift_status()
        if not drift_status:
            return
        
        num_drifted = drift_status["num_drifted"]
        total_cols = drift_status["total_cols"]
        drift_ratio = drift_status["drift_ratio"]
        
        print(f"\n📊 Drift Analysis:")
        print(f"   Total Columns: {total_cols}")
        print(f"   Drifted Columns: {num_drifted}")
        print(f"   Drift Ratio: {drift_ratio:.2%}")
        print(f"   Threshold: {DRIFT_THRESHOLD:.0%}")
        
        if drift_ratio >= DRIFT_THRESHOLD:
            print(f"\n⚠️  Drift exceeds threshold!")
            print("📄 Review drift report:")
            print("➡  reports/drift_report.html")
            
            decision = input("\n🤔 Approve retraining? (y/n/auto): ").strip().lower()
            
            if decision == "y":
                approve_retraining(silent=False)
            elif decision == "auto":
                auto_resolve_drift(silent=False)
            else:
                print("✋ Retraining rejected by human")
            
            # Ask if user wants to continue
            continue_decision = input("\n🔄 Check drift again? (y/n): ").strip().lower()
            if continue_decision != "y":
                break
        else:
            print("\n✅ Drift acceptable — no action needed")
            
            # Offer reset option
            reset_decision = input("\n💡 Want to reset and simulate high drift for demo? (y/n): ").strip().lower()
            
            if reset_decision == "y":
                result = reset_demo(silent=False)
                if result["success"]:
                    print("🔄 Re-running drift detection...\n")
                    # Loop continues to re-check drift
                else:
                    break
            else:
                break  # Exit if no reset

if __name__ == "__main__":
    run_pipeline()
