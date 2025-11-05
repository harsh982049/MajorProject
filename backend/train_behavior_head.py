import pandas as pd
import numpy as np
import joblib
import json
from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from lightgbm import LGBMRegressor
import warnings

warnings.filterwarnings('ignore')

# --- 1. DEFINE PATHS ---
# Assumes this script is run from the `backend` directory
SCRIPT_DIR = Path(__file__).resolve().parent
ARTIFACTS_DIR = SCRIPT_DIR / "artifacts"
LABELS_DIR = SCRIPT_DIR / "labels"
DATA_FILE = LABELS_DIR / "stress_30s.csv"

# --- 2. DEFINE FEATURES AND TARGET ---
# These are the 17 features from your global_head_meta.json and React app
FEATURES = [
    "ks_event_count",
    "ks_keydowns",
    "ks_keyups",
    "ks_unique_keys",
    "ks_mean_dwell_ms",
    "ks_median_dwell_ms",
    "ks_p95_dwell_ms",
    "ks_mean_ikg_ms",
    "ks_median_ikg_ms",
    "ks_p95_ikg_ms",
    "mouse_move_count",
    "mouse_click_count",
    "mouse_scroll_count",
    "mouse_total_distance_px",
    "mouse_mean_speed_px_s",
    "mouse_max_speed_px_s",
    "active_seconds_fraction"
]
TARGET = "stress_prob" # This is a float, so we use regression

# --- 3. LOAD AND PREPARE DATA ---
print(f"Loading data from {DATA_FILE}...")
if not DATA_FILE.exists():
    print(f"Error: Data file not found at {DATA_FILE}")
    print("Please ensure 'stress_30s.csv' is in the 'labels' folder.")
    exit()

df = pd.read_csv(DATA_FILE)
print(f"Loaded {len(df)} total rows.")

# Clean data based on logic from your service (confidence and coverage)
if "confident" in df.columns:
    df = df[df["confident"] == 1]
if "coverage" in df.columns:
    # Using 0.3 as a reasonable default from your service file
    df = df[df["coverage"] >= 0.3]

# Drop rows where target or features are missing
df = df.dropna(subset=FEATURES + [TARGET])
print(f"Filtered down to {len(df)} high-confidence rows.")

if len(df) < 100:
    print("Warning: Very little data (< 100 rows) for training. Model may be unreliable.")
    if len(df) == 0:
        print("Error: No data left after filtering. Aborting.")
        exit()

X = df[FEATURES].astype(np.float32)
y = df[TARGET].astype(np.float32)

# --- 4. TRAIN/VALIDATION SPLIT ---
X_train, X_val, y_train, y_val = train_test_split(
    X, y, test_size=0.2, random_state=42
)
print(f"Training with {len(X_train)} samples, validating with {len(X_val)} samples.")

# --- 5. TRAIN SCALER ---
# Your service loads the scaler separately, so we do the same.
print("Training StandardScaler...")
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_val_scaled = scaler.transform(X_val)

# --- 6. TRAIN MODEL (LGBMRegressor) ---
print("Training LGBMRegressor...")
# These are good starting parameters for a regression task
model = LGBMRegressor(
    n_estimators=200,
    learning_rate=0.05,
    num_leaves=31,
    random_state=42,
    n_jobs=-1
)
model.fit(X_train_scaled, y_train)

# --- 7. EVALUATE MODEL ---
print("Evaluating model on validation set...")
y_pred = model.predict(X_val_scaled)

mse = mean_squared_error(y_val, y_pred)
mae = mean_absolute_error(y_val, y_pred)
r2 = r2_score(y_val, y_pred)

print(f"--- Validation Metrics ---")
print(f"Mean Squared Error (MSE): {mse:.6f}")
print(f"Mean Absolute Error (MAE):  {mae:.6f}")
print(f"R-squared (R2):           {r2:.6f}")
print("----------------------------")

# --- 8. SAVE ARTIFACTS ---
# Ensure artifacts directory exists
# ARTIFACTS_DIR.mkdir(exist_ok=True)

# 8a. Save Scaler
scaler_path = ARTIFACTS_DIR / "global_head_scaler_2.joblib"
joblib.dump(scaler, scaler_path)
print(f"Scaler saved to {scaler_path}")

# 8b. Save Model
model_path = ARTIFACTS_DIR / "global_head_regression_behavior_2.joblib"
joblib.dump(model, model_path)
print(f"Model saved to {model_path}")

# 8c. Save Metadata (CRITICAL for your service to load the model)
meta_path = ARTIFACTS_DIR / "global_head_meta_2.json"
meta_data = {
    "source": "behavior_csv_only", # Tells service to use these 17 features
    "dims": len(FEATURES),
    "feature_names": FEATURES,
    "scaler": "StandardScaler(mean_, scale_)", # String representation
    "mode": "regression", # Tells service to call .predict()
    "model_name": "LGBMRegressor",
    "val_mse": mse,
    "val_mae": mae,
    "val_r2": r2,
    # Copying other keys from your original meta file
    "val_auc": None,
    "val_ap": None,
    "best_thresh": 0.5, # Default, can be tuned
    "best_f1": None,
    "y_bin_low": 0.3,   # From your original file
    "y_bin_high": 0.6   # From your original file
}

with open(meta_path, 'w') as f:
    json.dump(meta_data, f, indent=2)
print(f"Metadata saved to {meta_path}")

print("\n--- Training Complete ---")
print("New model, scaler, and metadata have been saved to the 'artifacts' folder.")
print("You can now restart your 'app.py' server to use the new model.")
