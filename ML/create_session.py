import pandas as pd
import numpy as np
from datetime import timedelta

# Step 1: Load the datasets
keystrokes_df = pd.read_csv("C:\\Users\\harsh\\OneDrive\\Desktop\\MajorProject\\ML\\keystrokes.tsv", sep='\t')
usercondition_df = pd.read_csv("C:\\Users\\harsh\\OneDrive\\Desktop\\MajorProject\\ML\\usercondition.tsv", sep='\t')
mouse_mov_speeds_df = pd.read_csv("C:\\Users\\harsh\\OneDrive\\Desktop\\MajorProject\\ML\\mouse_mov_speeds.tsv", sep='\t')
mousedata_df = pd.read_csv("C:\\Users\\harsh\\OneDrive\\Desktop\\MajorProject\\ML\\mousedata.tsv", sep='\t')

keystrokes_df['Press_Time'] = pd.to_datetime(keystrokes_df['Press_Time'], errors='coerce')
keystrokes_df['Relase_Time'] = pd.to_datetime(keystrokes_df['Relase_Time'], errors='coerce')
mouse_mov_speeds_df['Time'] = pd.to_datetime(mouse_mov_speeds_df['Time'], errors='coerce')
mousedata_df['Time'] = pd.to_datetime(mousedata_df['Time'], errors='coerce')
usercondition_df['Time'] = pd.to_datetime(usercondition_df['Time'], errors='coerce')

# print(keystrokes_df.dtypes)

# Create 1-minute session intervals
start_time = min(
    keystrokes_df['Press_Time'].min(),
    mousedata_df['Time'].min(),
    mouse_mov_speeds_df['Time'].min()
).floor('min')

end_time = max(
    keystrokes_df['Relase_Time'].max(),
    mousedata_df['Time'].max(),
    mouse_mov_speeds_df['Time'].max()
).ceil('min')

session_starts = pd.date_range(start=start_time, end=end_time, freq='30s')
session_intervals = [(t, t + timedelta(minutes=1)) for t in session_starts]

# Extract features per session
session_features = []

for start, end in session_intervals:
    session_data = {}
    session_data["session_start"] = start

    # ------------------ KEYSTROKES ------------------
    keys = keystrokes_df[(keystrokes_df["Press_Time"] >= start) & (keystrokes_df["Press_Time"] < end)]
    if not keys.empty:
        durations = (keys["Relase_Time"] - keys["Press_Time"]).dt.total_seconds()
        session_data["avg_keypress_duration"] = durations.mean()
        session_data["keypress_count"] = len(keys)
        session_data["backspace_count"] = (keys["Key"].str.lower() == 'backspace').sum()
        session_data["error_rate"] = session_data["backspace_count"] / session_data["keypress_count"]
        session_data["session_active"] = 1
    else:
        session_data["avg_keypress_duration"] = 0
        session_data["keypress_count"] = 0
        session_data["backspace_count"] = 0
        session_data["error_rate"] = 0
        session_data["session_active"] = 0

    # ------------------ MOUSE SPEED ------------------
    speed = mouse_mov_speeds_df[(mouse_mov_speeds_df["Time"] >= start) & (mouse_mov_speeds_df["Time"] < end)]
    session_data["avg_mouse_speed"] = speed["Speed(ms)"].mean() if not speed.empty else 0

    # ------------------ MOUSE EVENTS ------------------
    mouse = mousedata_df[(mousedata_df["Time"] >= start) & (mousedata_df["Time"] < end)]
    if not mouse.empty:
        session_data["mouse_move_count"] = (mouse["Event_Type"] == "Move").sum()
        session_data["mouse_click_count"] = (mouse["Event_Type"] == "Click").sum()
    else:
        session_data["mouse_move_count"] = 0
        session_data["mouse_click_count"] = 0

    # ------------------ CONTEXTUAL FEATURES ------------------
    session_data["hour"] = start.hour
    session_data["day_of_week"] = start.weekday()
    session_data["daylight_morning"] = 1 if 6 <= start.hour < 12 else 0
    session_data["daylight_evening"] = 1 if 17 <= start.hour < 21 else 0

    # ------------------ STRESS LABEL (usercondition) ------------------
    # Find closest stress label within ±1 minute
    condition_window = usercondition_df[
        # (usercondition_df["Time"] >= start - timedelta(minutes=1)) &
        # (usercondition_df["Time"] <= end + timedelta(minutes=1))
        (usercondition_df["Time"] >= start - timedelta(seconds=30)) &
        (usercondition_df["Time"] <= end + timedelta(seconds=30))
    ]
    if not condition_window.empty:
        closest_row = condition_window.iloc[(condition_window["Time"] - start).abs().argsort()[:1]]
        label = closest_row["Stress_Val"].values[0]
    else:
        label = np.nan

    session_data["stress_label"] = label

    session_features.append(session_data)

# Final DataFrame
features_df = pd.DataFrame(session_features)

# Drop rows with missing labels
features_df = features_df.dropna(subset=["stress_label"])

# Save to CSV for training
features_df.to_csv("session_features_30s.csv", index=False)
print("✅ Session-wise features saved to 'session_features_30s.csv'")