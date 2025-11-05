# train_hybrid_head.py
# Trains a hybrid global head: [mouse_embed ⊕ key_embed ⊕ 17_features] -> stress_prob
# Saves:
#   artifacts/global_head_scaler_hybrid.joblib
#   artifacts/global_head_regression_hybrid.joblib
#   artifacts/global_head_meta.json   # updated with "input_kind": "hybrid_embed"

from __future__ import annotations
import json
from pathlib import Path
import numpy as np
import pandas as pd
import joblib
import warnings

from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_squared_error, mean_absolute_error, roc_auc_score, average_precision_score

# Optional GBMs (preferred)
_LGBM_OK = True
try:
    from lightgbm import LGBMRegressor
except Exception:
    _LGBM_OK = False

_XGB_OK = True
try:
    from xgboost import XGBRegressor
except Exception:
    _XGB_OK = False


# ---------- Paths ----------
PROJECT_ROOT = Path(__file__).resolve().parent
ART = PROJECT_ROOT / "artifacts"
LAB = PROJECT_ROOT / "labels"

META_JSON  = ART / "global_head_meta.json"
MOUSE_EMB_30 = ART / "embeddings_mouse_pooled30.npz"    # pooled to 30s
KEY_EMB     = ART / "embeddings_keystroke.npz"
CSV         = LAB / "stress_30s.csv"

SCALER_OUT  = ART / "global_head_scaler_hybrid.joblib"
MODEL_OUT   = ART / "global_head_regression_hybrid.joblib"


# ---------- Helpers ----------
def _is_numeric(a: np.ndarray) -> bool:
    return np.issubdtype(a.dtype, np.number)

def _first_numeric_matrix_from_npz(p: Path) -> np.ndarray:
    """
    Load the first 2-D numeric array from an NPZ.
    Many embedding dumps include extra arrays like user_id, timestamps, etc.
    We scan keys and pick the first [N, D] numeric candidate.
    """
    if not p.exists():
        raise FileNotFoundError(p)
    d = np.load(p, allow_pickle=True)
    # Prefer common embedding keys first
    preferred = ["X", "embeddings", "mouse_embeddings", "key_embeddings", "E", "arr_0", "data"]
    tried = []
    for k in preferred + list(d.files):
        if k not in d.files:
            continue
        arr = d[k]
        tried.append(f"{k}:{arr.dtype}:{getattr(arr, 'shape', None)}")
        if isinstance(arr, np.ndarray) and arr.ndim == 2 and _is_numeric(arr):
            return arr.astype(np.float32, copy=False)
        # sometimes embeddings are [N, T, D] pooled later; take mean over T
        if isinstance(arr, np.ndarray) and arr.ndim == 3 and _is_numeric(arr):
            return arr.mean(axis=1).astype(np.float32, copy=False)
    keys = ", ".join(tried) if tried else "(no arrays?)"
    raise ValueError(
        f"No 2-D numeric matrix found in {p}.\n"
        f"Found arrays: {keys}\n"
        f"Tip: ensure the file contains a numeric [N, D] or [N, T, D] array."
    )

def _read_meta_and_features(meta_path: Path) -> tuple[list[str], dict]:
    if not meta_path.exists():
        raise FileNotFoundError(f"Missing {meta_path}")
    meta = json.loads(meta_path.read_text())
    feat_names = meta.get("feature_names")
    if not isinstance(feat_names, list) or not feat_names:
        raise ValueError("feature_names missing/invalid in global_head_meta.json")
    return feat_names, meta

def _extreme_mask(y: np.ndarray, lo: float = 0.30, hi: float = 0.70):
    lo_m = y <= lo
    hi_m = y >= hi
    return lo_m, hi_m


def main():
    print("== Hybrid Trainer: embeddings + 17 handcrafted features ==")

    # 1) Load meta & feature list
    feature_names, meta = _read_meta_and_features(META_JSON)
    print(f"Loaded {len(feature_names)} handcrafted features from meta.")

    # 2) Load embeddings robustly
    key_emb = _first_numeric_matrix_from_npz(KEY_EMB)           # [N, Dk]
    mouse_emb_30 = _first_numeric_matrix_from_npz(MOUSE_EMB_30) # [N, Dm]
    print(f"Key emb shape:   {key_emb.shape}")
    print(f"Mouse emb shape: {mouse_emb_30.shape}")

    # 3) Load labels CSV
    if not CSV.exists():
        raise FileNotFoundError(f"Missing labels CSV at {CSV}")
    df = pd.read_csv(CSV)

    # Validate features & label
    missing_feats = [f for f in feature_names if f not in df.columns]
    if missing_feats:
        raise ValueError(f"Missing feature columns in CSV: {missing_feats[:5]} ...")
    if "stress_prob" not in df.columns:
        if "label" in df.columns:
            df = df.rename(columns={"label": "stress_prob"})
        else:
            raise ValueError("CSV must contain 'stress_prob' column in [0..1].")

    # 4) Align lengths among [key_emb, mouse_emb_30, df]
    n_all = min(len(df), key_emb.shape[0], mouse_emb_30.shape[0])
    if n_all < 20:
        warnings.warn("Very few rows after alignment; metrics may be noisy.")
    df = df.iloc[:n_all].reset_index(drop=True)
    key_emb = key_emb[:n_all]
    mouse_emb_30 = mouse_emb_30[:n_all]

    # 5) Build X (hybrid) and y
    feats = df[feature_names].astype(np.float32).values      # [N, 17]
    X = np.concatenate([mouse_emb_30, key_emb, feats], axis=1)  # [N, Dm+Dk+17]
    y = df["stress_prob"].astype(np.float32).values          # [N]

    print(f"Hybrid X shape: {X.shape}, y shape: {y.shape}")

    # 6) Train/Val split (time-aware: last 20% as validation)
    n = len(y)
    split_idx = int(0.8 * n) if n >= 10 else max(1, n - 2)
    X_train, X_val = X[:split_idx], X[split_idx:]
    y_train, y_val = y[:split_idx], y[split_idx:]

    # 7) Scaler (kept for service contract; GBMs don't need it but service calls scaler.transform)
    scaler = StandardScaler(with_mean=True, with_std=True)
    X_train_z = scaler.fit_transform(X_train)
    X_val_z   = scaler.transform(X_val) if len(X_val) else np.zeros((0, X_train.shape[1]), dtype=np.float32)

    # 8) Pick model: LGBM → XGB → Ridge (fallbacks)
    if _LGBM_OK:
        model = LGBMRegressor(
            n_estimators=800,
            learning_rate=0.03,
            max_depth=-1,
            num_leaves=63,
            subsample=0.9,
            colsample_bytree=0.9,
            reg_alpha=0.0,
            reg_lambda=1.0,
            random_state=42,
        )
        model_name = "LGBMRegressor"
    elif _XGB_OK:
        model = XGBRegressor(
            n_estimators=900,
            learning_rate=0.03,
            max_depth=7,
            subsample=0.9,
            colsample_bytree=0.9,
            reg_lambda=1.0,
            objective="reg:squarederror",
            tree_method="hist",
            random_state=42,
        )
        model_name = "XGBRegressor"
    else:
        model = Ridge(alpha=1.0)
        model_name = "Ridge (fallback)"
        warnings.warn("LightGBM/XGBoost not found – falling back to Ridge.")

    print(f"Training head: {model_name}")
    model.fit(X_train_z, y_train)

    # 9) Evaluate
    def _clip01(a): return np.clip(a, 0, 1)
    y_hat_tr = _clip01(model.predict(X_train_z))
    y_hat_va = _clip01(model.predict(X_val_z)) if len(X_val_z) else np.array([])

    def _reg_report(yt, yp, tag):
        if len(yt) == 0:
            print(f"[{tag}] (no samples)")
            return
        mse = mean_squared_error(yt, yp)
        mae = mean_absolute_error(yt, yp)
        print(f"[{tag}] MSE={mse:.5f}  MAE={mae:.5f}")

    _reg_report(y_train, y_hat_tr, "train")
    _reg_report(y_val,   y_hat_va, "valid")

    # AUC/PR on extremes (y<=0.30 vs y>=0.70)
    def _bin_metrics(yt, yp, tag):
        if len(yt) == 0:
            print(f"[{tag} extremes] (no samples)")
            return
        lo = yt <= 0.30
        hi = yt >= 0.70
        mask = lo | hi
        if mask.sum() >= 10:
            yb = (yt >= 0.70).astype(int)[mask]
            pb = yp[mask]
            try:
                auc = roc_auc_score(yb, pb)
            except ValueError:
                auc = float("nan")
            try:
                ap  = average_precision_score(yb, pb)
            except ValueError:
                ap = float("nan")
            print(f"[{tag} extremes] N={mask.sum()}  ROC-AUC={auc:.4f}  PR-AUC={ap:.4f}")
        else:
            print(f"[{tag} extremes] Not enough extreme samples (N={mask.sum()}).")

    _bin_metrics(y_train, y_hat_tr, "train")
    _bin_metrics(y_val,   y_hat_va, "valid")

    # 10) Save artifacts
    ART.mkdir(parents=True, exist_ok=True)
    joblib.dump(scaler, SCALER_OUT)
    joblib.dump(model,  MODEL_OUT)
    print(f"Saved scaler  → {SCALER_OUT}")
    print(f"Saved model   → {MODEL_OUT}")

    # 11) Update meta for hybrid mode (non-destructive: keep your thresholds/feature_names)
    meta_backup = META_JSON.with_suffix(".json.bak")
    if META_JSON.exists():
        meta_backup.write_text(META_JSON.read_text())

    meta["mode"] = "regression"
    meta["input_kind"] = "hybrid_embed"
    meta["hybrid_dims"] = {
        "mouse_embed": int(mouse_emb_30.shape[1]),
        "key_embed": int(key_emb.shape[1]),
        "features": len(feature_names),
        "total": int(mouse_emb_30.shape[1] + key_emb.shape[1] + len(feature_names)),
    }
    meta["best_thresh"] = float(meta.get("best_thresh", 0.5))
    META_JSON.write_text(json.dumps(meta, indent=2))
    print(f"Updated meta  → {META_JSON} (backup at {meta_backup})")
    print("Done.")


if __name__ == "__main__":
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        main()
