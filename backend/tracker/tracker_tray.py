import time
import csv
import threading
import os
import signal
import psutil
import sys
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime, timedelta
from pynput import keyboard, mouse
from plyer import notification
from pystray import Icon, MenuItem, Menu
from PIL import Image, ImageDraw
import pandas as pd
import tkinter as tk
from tkinter import ttk
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.figure import Figure
import pickle  # Added for loading the ML model

# Global State Variables
keystrokes = []
backspace_count = 0
mouse_movements = []
key_press_times = []  # Timing between key presses
mouse_click_count = 0  # Count mouse clicks
idle_time = 0  # Track idle time
last_action_time = time.time()  # Track last action time
running = True
paused = False

# Process management
pid_file = "tracker_tray.pid"

# Paths to the saved model and scaler
model_path = "C:\\Users\\harsh\\OneDrive\\Desktop\\MajorProject\\ML\\rf_model_30s.pkl"
scaler_path = "C:\\Users\\harsh\\OneDrive\\Desktop\\MajorProject\\ML\\scaler_30s.pkl"

# Logging Setup
log_file_path = "stress_log2.csv"

# Load the model and scaler
try:
    with open(model_path, 'rb') as f:
        stress_model = pickle.load(f)
    
    with open(scaler_path, 'rb') as f:
        scaler = pickle.load(f)
    
    model_loaded = True
    print("✅ ML model and scaler loaded successfully")
except Exception as e:
    model_loaded = False
    print(f"❌ Error loading ML model: {e}")

# Feature columns used for prediction (must match what the model was trained on)
feature_columns = [
    'avg_keypress_duration', 'keypress_count', 'backspace_count', 'error_rate',
    'avg_mouse_speed', 'mouse_move_count', 'mouse_click_count',
    'hour', 'day_of_week', 'daylight_morning', 'daylight_evening', 'session_active'
]

# Create the log file if it doesn't exist
if not os.path.exists(log_file_path):
    with open(log_file_path, mode='w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow([
            "timestamp", "typing_speed", "backspace_rate", "mouse_jitter", 
            "key_rhythm_consistency", "idle_time", "mouse_click_rate", 
            "stress_level", "stress_factors", "ml_prediction"  # Added ML prediction column
        ])
        
# ML prediction log file (to track all features and predictions)
ml_log_file_path = "ml_predictions.csv"
if not os.path.exists(ml_log_file_path):
    with open(ml_log_file_path, mode='w', newline='') as f:
        writer = csv.writer(f)
        header = ["timestamp", "ml_prediction"] + feature_columns
        writer.writerow(header)


# Process Management
def check_existing_process():
    """Check if another instance is running and terminate it if found"""
    if os.path.exists(pid_file):
        try:
            with open(pid_file, 'r') as f:
                old_pid = int(f.read().strip())
            
            # Check if the process still exists
            if psutil.pid_exists(old_pid):
                try:
                    process = psutil.Process(old_pid)
                    if "python" in process.name().lower():
                        print(f"Terminating existing process with PID {old_pid}")
                        process.terminate()
                        process.wait(timeout=3)
                except Exception as e:
                    print(f"Error terminating existing process: {e}")
        except Exception as e:
            print(f"Error reading PID file: {e}")
    
    # Write current PID
    with open(pid_file, 'w') as f:
        f.write(str(os.getpid()))

def cleanup():
    """Clean up resources before exiting"""
    if os.path.exists(pid_file):
        try:
            os.remove(pid_file)
        except:
            pass

# -----------------------
# Helper Functions
# -----------------------
def calculate_typing_speed():
    """Calculate typing speed in keystrokes per second"""
    if len(keystrokes) < 2:
        return 0
    intervals = [t2 - t1 for t1, t2 in zip(keystrokes, keystrokes[1:]) if t2 - t1 < 5]
    return round(len(intervals) / (sum(intervals) + 1e-5), 2) if intervals else 0

def calculate_backspace_rate():
    """Calculate the rate of backspace usage"""
    total_keys = len(keystrokes)
    return round(backspace_count / total_keys, 2) if total_keys > 0 else 0

def calculate_mouse_jitter():
    """Calculate mouse movement instability"""
    if len(mouse_movements) < 2:
        return 0
    dist = 0
    for (x1, y1), (x2, y2) in zip(mouse_movements, mouse_movements[1:]):
        dist += ((x2 - x1)**2 + (y2 - y1)**2) ** 0.5
    return round(dist / len(mouse_movements), 2)

def calculate_key_rhythm_consistency():
    """Calculate consistency in typing rhythm"""
    if len(key_press_times) < 3:
        return 1.0  # Default value (consistent)
    
    # Calculate variance in key press intervals
    intervals = [t2 - t1 for t1, t2 in zip(key_press_times, key_press_times[1:])]
    if not intervals:
        return 1.0
    
    # Calculate coefficient of variation (normalized measure of dispersion)
    mean_interval = sum(intervals) / len(intervals)
    if mean_interval == 0:
        return 0.0
    
    std_dev = (sum((x - mean_interval) ** 2 for x in intervals) / len(intervals)) ** 0.5
    coef_variation = std_dev / mean_interval
    
    # Normalize to 0-1 where 1 is very consistent (low variation) and 0 is inconsistent
    consistency = max(0, min(1, 1 - (coef_variation / 2)))
    return round(consistency, 2)

def calculate_mouse_click_rate():
    """Calculate mouse click frequency"""
    total_time = 30  # 30 seconds monitoring period
    return round(mouse_click_count / total_time, 2)

def calculate_idle_time():
    """Calculate idle time in seconds"""
    return round(idle_time, 1)

def detect_stress(metrics):
    """
    Enhanced stress detection using multiple parameters
    Returns a tuple of (stress_level, factors_list)
    Stress level is on a scale of 0-10 where:
    0-3: Calm
    4-6: Moderate stress
    7-10: High stress
    """
    ts = metrics['typing_speed']
    br = metrics['backspace_rate']
    mj = metrics['mouse_jitter']
    kr = metrics['key_rhythm']
    it = metrics['idle_time']
    mc = metrics['mouse_clicks']
    
    stress_score = 0
    factors = []
    
    # Typing speed analysis
    if ts < 0.5:
        stress_score += 2
        factors.append("very slow typing")
    elif ts < 1.2:
        stress_score += 1
        factors.append("slow typing")
    elif ts > 4:
        stress_score += 1
        factors.append("unusually fast typing")
    
    # Backspace rate analysis
    if br > 0.3:
        stress_score += 2
        factors.append("high error correction")
    elif br > 0.15:
        stress_score += 1
        factors.append("moderate error correction")
    
    # Mouse jitter analysis
    if mj > 1000:
        stress_score += 2
        factors.append("excessive mouse movement")
    elif mj > 600:
        stress_score += 1
        factors.append("jittery mouse movement")
    
    # Typing rhythm consistency
    if kr < 0.4:
        stress_score += 2
        factors.append("erratic typing pattern")
    elif kr < 0.6:
        stress_score += 1
        factors.append("inconsistent typing rhythm")
    
    # Idle time analysis (frequent short pauses may indicate stress)
    if 3 < it < 10:
        stress_score += 1
        factors.append("frequent brief pauses")
    
    # Mouse click analysis
    if mc > 2:
        stress_score += 1
        factors.append("excessive clicking")
    
    # Cap the stress score at 10
    stress_score = min(stress_score, 10)
    
    # Determine stress level category
    if stress_score <= 3:
        level = "CALM"
    elif stress_score <= 6:
        level = "MODERATE STRESS"
    else:
        level = "HIGH STRESS"
    
    return (level, stress_score, factors)

def predict_stress_with_ml(data):
    """
    Predict stress using the loaded ML model
    Returns 1 for stressed, 0 for not stressed
    """
    if not model_loaded:
        return None  # Return None if model failed to load
    
    try:
        # Create a DataFrame with the expected features
        features_df = pd.DataFrame([data], columns=feature_columns)
        
        # Scale the features if the model was trained on scaled data
        # For RandomForest, scaling is not necessary but other models may need it
        # features_scaled = scaler.transform(features_df)
        
        # Make prediction
        prediction = stress_model.predict(features_df)[0]
        
        # Get probability for the "stressed" class (class 1)
        # probability = stress_model.predict_proba(features_df)[0][1]
        
        return prediction
    except Exception as e:
        print(f"Error making prediction: {e}")
        return None

def collect_features_for_ml():
    """
    Collect and format features for the ML model.
    These must match the feature columns the model was trained on.
    """
    # Get the current time
    now = datetime.now()
    
    # Calculate metrics for ML prediction
    avg_keypress_dur = 0
    if len(key_press_times) >= 2:
        intervals = [t2 - t1 for t1, t2 in zip(key_press_times, key_press_times[1:])]
        if intervals:
            avg_keypress_dur = sum(intervals) / len(intervals)
    
    # Count keypresses
    keypresses = len(keystrokes)
    
    # Error rate (backspace / total keypresses)
    error_rate = backspace_count / keypresses if keypresses > 0 else 0
    
    # Average mouse speed
    avg_mouse_speed = calculate_mouse_jitter()  # Using jitter as a proxy for speed
    
    # Mouse movement count
    mouse_move_count = len(mouse_movements)
    
    # Check if session is active (any keyboard or mouse activity)
    session_active = 1 if keypresses > 0 or mouse_move_count > 0 or mouse_click_count > 0 else 0
    
    # Time features
    hour = now.hour
    day_of_week = now.weekday()  # 0 = Monday, 6 = Sunday
    
    # Daylight indicators (simplified, adjust as needed)
    daylight_morning = 1 if 6 <= hour <= 11 else 0
    daylight_evening = 1 if 18 <= hour <= 21 else 0
    
    # Create feature dictionary
    features = {
        'avg_keypress_duration': avg_keypress_dur,
        'keypress_count': keypresses,
        'backspace_count': backspace_count,
        'error_rate': error_rate,
        'avg_mouse_speed': avg_mouse_speed,
        'mouse_move_count': mouse_move_count,
        'mouse_click_count': mouse_click_count,
        'hour': hour,
        'day_of_week': day_of_week,
        'daylight_morning': daylight_morning,
        'daylight_evening': daylight_evening,
        'session_active': session_active
    }
    
    return features

def show_popup(status, score, reasons, ml_prediction=None):
    """Display a notification with stress status"""
    if not reasons:
        message = "You're doing fine. Keep going!"
    else:
        message = f"Stress level {score}/10: {', '.join(reasons)}"
    
    # Add ML prediction if available
    if ml_prediction is not None:
        ml_status = "STRESSED" if ml_prediction == 1 else "NOT STRESSED"
        message += f"\nML Prediction: {ml_status}"
    
    try:
        # Try different notification approaches
        try:
            # First attempt with plyer
            notification.notify(
                title=f"Stress Status: {status}",
                message=message,
                app_name="Stress Tracker",
                timeout=5
            )
        except Exception as e:
            print(f"Plyer notification error: {e}")
            # Fallback to a simple Tkinter popup
            show_tkinter_popup(status, message)
    except Exception as e:
        print(f"All notification methods failed: {e}")

def show_tkinter_popup(title, message):
    """Fallback notification using Tkinter"""
    try:
        popup = tk.Tk()
        popup.title(title)
        popup.attributes("-topmost", True)
        
        # Calculate position (bottom right corner)
        screen_width = popup.winfo_screenwidth()
        screen_height = popup.winfo_screenheight()
        popup.geometry(f"300x100+{screen_width-320}+{screen_height-150}")
        
        # Add content
        ttk.Label(popup, text=message, wraplength=280).pack(pady=10, padx=10, expand=True)
        ttk.Button(popup, text="OK", command=popup.destroy).pack(pady=5)
        
        # Auto-close after 5 seconds
        popup.after(5000, popup.destroy)
        
        popup.mainloop()
    except Exception as e:
        print(f"Tkinter popup error: {e}")


# Event Handlers
def on_press(key):
    if paused or not running:
        return
    
    global backspace_count, last_action_time, idle_time
    
    # Update idle time
    current_time = time.time()
    if last_action_time:
        idle_duration = current_time - last_action_time
        if idle_duration > 1:  # Only count idle periods > 1 second
            idle_time += idle_duration
    
    last_action_time = current_time
    keystrokes.append(current_time)
    key_press_times.append(current_time)
    
    if key == keyboard.Key.backspace:
        backspace_count += 1

def on_click(x, y, button, pressed):
    if paused or not running:
        return
    
    global mouse_click_count, last_action_time, idle_time
    
    if pressed:
        # Update idle time
        current_time = time.time()
        if last_action_time:
            idle_duration = current_time - last_action_time
            if idle_duration > 1:
                idle_time += idle_duration
        
        last_action_time = current_time
        mouse_click_count += 1

def on_move(x, y):
    if paused or not running:
        return
    
    global last_action_time, idle_time
    
    # Update idle time
    current_time = time.time()
    if last_action_time:
        idle_duration = current_time - last_action_time
        if idle_duration > 1:
            idle_time += idle_duration
    
    last_action_time = current_time
    mouse_movements.append((x, y))
    if len(mouse_movements) > 100:
        mouse_movements.pop(0)


# Main Tracking Logic
def monitor_behavior():
    while running:
        if not paused and running:
            try:
                # Calculate metrics
                ts = calculate_typing_speed()
                br = calculate_backspace_rate()
                mj = calculate_mouse_jitter()
                kr = calculate_key_rhythm_consistency()
                it = calculate_idle_time()
                mc = calculate_mouse_click_rate()
                
                # Package metrics for stress detection
                metrics = {
                    'typing_speed': ts,
                    'backspace_rate': br,
                    'mouse_jitter': mj,
                    'key_rhythm': kr,
                    'idle_time': it,
                    'mouse_clicks': mc
                }
                
                # Detect stress using rule-based method
                status, score, factors = detect_stress(metrics)
                
                # Collect features for ML prediction
                ml_features = collect_features_for_ml()
                
                # Make ML prediction if model is loaded
                ml_prediction = predict_stress_with_ml(ml_features) if model_loaded else None
                
                # Show notification with both rule-based and ML results
                show_popup(status, score, factors, ml_prediction)

                # Current timestamp
                timestamp = time.time()
                
                # Log data to the main log file
                with open(log_file_path, mode='a', newline='') as f:
                    writer = csv.writer(f)
                    writer.writerow([
                        timestamp, ts, br, mj, kr, it, mc, score, 
                        ','.join(factors) if factors else "none",
                        int(ml_prediction) if ml_prediction is not None else "N/A"
                    ])
                
                # Log data to the ML-specific log file
                if model_loaded:
                    with open(ml_log_file_path, mode='a', newline='') as f:
                        writer = csv.writer(f)
                        ml_data = [timestamp, int(ml_prediction)]
                        for feature in feature_columns:
                            ml_data.append(ml_features[feature])
                        writer.writerow(ml_data)

                # Reset counters
                keystrokes.clear()
                key_press_times.clear()
                mouse_movements.clear()
                global backspace_count, mouse_click_count, idle_time
                backspace_count = 0
                mouse_click_count = 0
                idle_time = 0
                
            except Exception as e:
                print(f"Error in monitor thread: {e}")

        time.sleep(30)  # Take measurements every 30 seconds
        
        # Break the loop if not running
        if not running:
            break

# Data Visualization
def show_graphs():
    """Display graphs of tracked metrics"""
    try:
        # Load data from CSV
        if not os.path.exists(log_file_path) or os.path.getsize(log_file_path) == 0:
            show_popup("No Data", "No tracking data available yet.", [])
            return
        
        df = pd.read_csv(log_file_path)
        if len(df) < 2:
            show_popup("Insufficient Data", "Need more data points for visualization.", [])
            return
        
        # Convert timestamp to datetime for better x-axis
        df['datetime'] = pd.to_datetime(df['timestamp'], unit='s')
        
        # Create visualization window
        root = tk.Tk()
        root.title("Stress Tracking Metrics")
        root.geometry("1000x800")
        
        # Create notebook (tabbed interface)
        notebook = ttk.Notebook(root)
        notebook.pack(fill='both', expand=True, padx=10, pady=10)
        
        # Tab 1: Stress Level Over Time
        tab1 = ttk.Frame(notebook)
        notebook.add(tab1, text="Stress Level")
        
        fig1 = Figure(figsize=(10, 6))
        ax1 = fig1.add_subplot(111)
        ax1.plot(df['datetime'], df['stress_level'], 'r-', linewidth=2, label='Rule-Based')
        
        # Plot ML predictions if available
        if 'ml_prediction' in df.columns and df['ml_prediction'].dtype != object:
            ml_data = df[df['ml_prediction'].notna()]
            if not ml_data.empty:
                # Convert binary predictions to a stress scale (0 to 10) for visualization
                ml_data['ml_stress_scale'] = ml_data['ml_prediction'] * 10
                ax1.plot(ml_data['datetime'], ml_data['ml_stress_scale'], 'b--', linewidth=1.5, 
                         label='ML Prediction (scaled)')
        
        ax1.set_title('Stress Level Over Time')
        ax1.set_xlabel('Time')
        ax1.set_ylabel('Stress Level (0-10)')
        ax1.grid(True)
        ax1.legend()
        
        # Add a horizontal line at moderate stress threshold
        ax1.axhline(y=4, color='orange', linestyle='--', alpha=0.7)
        ax1.axhline(y=7, color='red', linestyle='--', alpha=0.7)
        ax1.text(df['datetime'].iloc[0], 3.5, 'Calm', color='green', fontsize=10)
        ax1.text(df['datetime'].iloc[0], 5.5, 'Moderate', color='orange', fontsize=10)
        ax1.text(df['datetime'].iloc[0], 8.5, 'High Stress', color='red', fontsize=10)
        
        canvas1 = FigureCanvasTkAgg(fig1, tab1)
        canvas1.draw()
        canvas1.get_tk_widget().pack(fill=tk.BOTH, expand=True)
        
        # Tab 2: Typing Metrics
        tab2 = ttk.Frame(notebook)
        notebook.add(tab2, text="Typing Metrics")
        
        fig2 = Figure(figsize=(10, 6))
        ax2 = fig2.add_subplot(111)
        ax2.plot(df['datetime'], df['typing_speed'], 'b-', label='Typing Speed')
        ax2.plot(df['datetime'], df['backspace_rate'], 'r-', label='Backspace Rate')
        ax2.plot(df['datetime'], df['key_rhythm_consistency'], 'g-', label='Rhythm Consistency')
        ax2.set_title('Typing Metrics Over Time')
        ax2.set_xlabel('Time')
        ax2.set_ylabel('Value')
        ax2.legend()
        ax2.grid(True)
        
        canvas2 = FigureCanvasTkAgg(fig2, tab2)
        canvas2.draw()
        canvas2.get_tk_widget().pack(fill=tk.BOTH, expand=True)
        
        # Tab 3: Mouse Metrics
        tab3 = ttk.Frame(notebook)
        notebook.add(tab3, text="Mouse Metrics")
        
        fig3 = Figure(figsize=(10, 6))
        ax3 = fig3.add_subplot(111)
        ax3.plot(df['datetime'], df['mouse_jitter'], 'b-', label='Mouse Jitter')
        ax3.plot(df['datetime'], df['mouse_click_rate'], 'r-', label='Click Rate')
        ax3.set_title('Mouse Activity Over Time')
        ax3.set_xlabel('Time')
        ax3.set_ylabel('Value')
        ax3.legend()
        ax3.grid(True)
        
        canvas3 = FigureCanvasTkAgg(fig3, tab3)
        canvas3.draw()
        canvas3.get_tk_widget().pack(fill=tk.BOTH, expand=True)
        
        # Tab 4: Stress Factors
        tab4 = ttk.Frame(notebook)
        notebook.add(tab4, text="Recent Stress Factors")
        
        # Get the most recent factors
        recent_data = df.tail(10).copy()
        recent_data['factors'] = recent_data['stress_factors'].fillna('none')
        
        # Create a text widget to show recent stress factors
        factor_text = tk.Text(tab4, wrap=tk.WORD, height=20, width=80)
        factor_text.pack(padx=10, pady=10, fill=tk.BOTH, expand=True)
        
        factor_text.insert(tk.END, "Recent Stress Factors:\n\n")
        for idx, row in recent_data.iterrows():
            timestamp = row['datetime'].strftime('%Y-%m-%d %H:%M:%S')
            level = row['stress_level']
            factors = row['factors']
            ml_pred = row.get('ml_prediction', 'N/A')
            
            factor_text.insert(tk.END, f"Time: {timestamp}\n")
            factor_text.insert(tk.END, f"Stress Level: {level}/10\n")
            
            # Add ML prediction if available
            if ml_pred != 'N/A':
                ml_status = "STRESSED" if ml_pred == 1 else "NOT STRESSED"
                factor_text.insert(tk.END, f"ML Prediction: {ml_status}\n")
                
            factor_text.insert(tk.END, f"Factors: {factors}\n\n")
            
            # Add tags for color coding
            if level <= 3:
                factor_text.tag_add("calm", f"{factor_text.index('end-4l')} linestart", f"{factor_text.index('end-4l')} lineend")
                factor_text.tag_config("calm", foreground="green")
            elif level <= 6:
                factor_text.tag_add("moderate", f"{factor_text.index('end-4l')} linestart", f"{factor_text.index('end-4l')} lineend")
                factor_text.tag_config("moderate", foreground="orange")
            else:
                factor_text.tag_add("high", f"{factor_text.index('end-4l')} linestart", f"{factor_text.index('end-4l')} lineend")
                factor_text.tag_config("high", foreground="red")
        
        factor_text.config(state=tk.DISABLED)
        
        # If ML log file exists, add a tab for ML predictions
        if os.path.exists(ml_log_file_path) and os.path.getsize(ml_log_file_path) > 0:
            try:
                ml_df = pd.read_csv(ml_log_file_path)
                if len(ml_df) > 0:
                    # Tab 5: ML Predictions
                    tab5 = ttk.Frame(notebook)
                    notebook.add(tab5, text="ML Predictions")
                    
                    # Convert timestamp to datetime
                    ml_df['datetime'] = pd.to_datetime(ml_df['timestamp'], unit='s')
                    
                    fig5 = Figure(figsize=(10, 6))
                    ax5 = fig5.add_subplot(111)
                    
                    # Plot the predictions (0 or 1) as points
                    ax5.plot(ml_df['datetime'], ml_df['ml_prediction'], 'bo-', markersize=6)
                    
                    # Set y-axis limits and ticks for binary predictions
                    ax5.set_ylim(-0.1, 1.1)
                    ax5.set_yticks([0, 1])
                    ax5.set_yticklabels(['Not Stressed', 'Stressed'])
                    
                    ax5.set_title('ML Stress Predictions Over Time')
                    ax5.set_xlabel('Time')
                    ax5.grid(True)
                    
                    canvas5 = FigureCanvasTkAgg(fig5, tab5)
                    canvas5.draw()
                    canvas5.get_tk_widget().pack(fill=tk.BOTH, expand=True)
            except Exception as e:
                print(f"Error creating ML predictions tab: {e}")
        
        # Add a close button at the bottom
        close_btn = ttk.Button(root, text="Close", command=root.destroy)
        close_btn.pack(pady=10)
        
        root.mainloop()
        
    except Exception as e:
        print(f"Error displaying graphs: {e}")
        show_popup("Error", f"Failed to display graphs: {str(e)}", [])


# Tray Icon UI
def create_image():
    """Create the tray icon image"""
    image = Image.new('RGB', (64, 64), color='navy')
    draw = ImageDraw.Draw(image)
    draw.ellipse((16, 16, 48, 48), fill='gold')
    return image

def exit_application(icon):
    """Properly exit the application"""
    global running
    running = False
    icon.stop()
    
    # Stop listeners and other threads
    if keyboard_listener.is_alive():
        keyboard_listener.stop()
    if mouse_listener.is_alive():
        mouse_listener.stop()
    
    # Cleanup and exit the application
    cleanup()
    os._exit(0)  # Force exit to ensure all threads terminate

def on_quit(icon, item):
    """Handle quit menu selection"""
    exit_application(icon)

def on_pause(icon, item):
    """Handle pause menu selection"""
    global paused
    paused = True
    show_popup("Paused", 0, ["Tracking paused from tray"])

def on_resume(icon, item):
    """Handle resume menu selection"""
    global paused
    paused = False
    show_popup("Resumed", 0, ["Tracking resumed"])

def on_show_graphs(icon, item):
    """Handle show graphs menu selection"""
    # Create a separate thread for the graphs to not block the tray icon
    graph_thread = threading.Thread(target=show_graphs)
    graph_thread.daemon = True
    graph_thread.start()


# Program Entrypoint
if __name__ == '__main__':
    # Check for existing instances and terminate them
    check_existing_process()
    
    # Set up signal handlers for clean shutdown
    signal.signal(signal.SIGINT, lambda sig, frame: cleanup())
    signal.signal(signal.SIGTERM, lambda sig, frame: cleanup())
    
    # Start listeners
    keyboard_listener = keyboard.Listener(on_press=on_press)
    mouse_listener = mouse.Listener(on_move=on_move, on_click=on_click)
    keyboard_listener.start()
    mouse_listener.start()

    # Start tracking thread
    tracking_thread = threading.Thread(target=monitor_behavior, daemon=True)
    tracking_thread.start()

    # Create and run tray icon
    tray_icon = Icon("StressTracker")
    tray_icon.icon = create_image()
    tray_icon.menu = Menu(
        MenuItem("Show Graphs", on_show_graphs),
        MenuItem("Pause Tracking", on_pause),
        MenuItem("Resume Tracking", on_resume),
        MenuItem("Quit", on_quit)
    )
    
    # Set handler to cleanup on exit
    tray_icon.run()
    
    # If we get here, the icon has stopped running
    running = False
    
    # Wait for threads to finish
    if tracking_thread.is_alive():
        tracking_thread.join(timeout=1)
    
    cleanup()