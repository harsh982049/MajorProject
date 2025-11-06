import os, json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.ensemble import GradientBoostingRegressor
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import (
    mean_absolute_error, mean_squared_error, r2_score,
    roc_auc_score, average_precision_score,
    f1_score, accuracy_score
)
from sklearn.utils.validation import check_is_fitted
from joblib import dump

# ----------------------- CONFIG -----------------------
CSV_PATH = "labels/stress_hybrid_10s.csv"  # change if needed

MVP_FIELDS = [
    "ks_event_count","ks_keydowns","ks_keyups","ks_unique_keys",
    "ks_mean_dwell_ms","ks_median_dwell_ms","ks_p95_dwell_ms",
    "ks_mean_ikg_ms","ks_median_ikg_ms","ks_p95_ikg_ms",
    "mouse_move_count","mouse_click_count","mouse_scroll_count",
    "mouse_total_distance_px","mouse_mean_speed_px_s","mouse_max_speed_px_s",
    "active_seconds_fraction"
]
EMB_MOUSE_PREFIX = "mouse_emb_"
EMB_KEY_PREFIX   = "key_emb_"

OUT_DIR = "models_personal"
os.makedirs(OUT_DIR, exist_ok=True)
FIG_DIR = os.path.join(OUT_DIR, "figs")
os.makedirs(FIG_DIR, exist_ok=True)

# ----------------------- DATA -------------------------
def load_clean_dataframe(path):
    if not os.path.exists(path):
        raise FileNotFoundError(f"CSV not found: {path}")
    import io
    with open(path, 'r', encoding='latin1', errors='ignore') as f:
        df = pd.read_csv(io.StringIO(f.read()))


    # Keep reliable supervision
    if "confident" in df:      df = df[df["confident"] == 1]
    if "n_face_frames" in df:  df = df[df["n_face_frames"].fillna(0) > 0]
    df = df[np.isfinite(df["stress_prob"])]
    if "pred_emotion" in df:
        df = df[df["pred_emotion"].notna() & (df["pred_emotion"].astype(str).str.len() > 0)]

    if "t1_unix" in df:
        df = df.sort_values("t1_unix").reset_index(drop=True)

    return df

def pick_columns(df):
    emb_mouse_cols = [c for c in df.columns if c.startswith(EMB_MOUSE_PREFIX)]
    emb_key_cols   = [c for c in df.columns if c.startswith(EMB_KEY_PREFIX)]
    emb_cols = emb_mouse_cols + emb_key_cols

    mvp_cols = [c for c in MVP_FIELDS if c in df.columns]
    mask_cols = [c for c in ["has_mouse_emb","has_keys_emb"] if c in df.columns]
    y_col = "stress_prob"
    return emb_cols, mvp_cols, mask_cols, y_col

def chrono_train_test_split(X, y, test_frac=0.2):
    n = len(y)
    test_n = max(1, int(round(test_frac * n)))
    train_n = n - test_n
    return (X.iloc[:train_n], X.iloc[train_n:],
            y.iloc[:train_n], y.iloc[train_n:])

# ----------------------- MODELS -----------------------
def make_gbrt(n_estimators=800):
    return GradientBoostingRegressor(
        loss="squared_error",
        n_estimators=n_estimators,
        learning_rate=0.03,
        max_depth=3,
        min_samples_leaf=3,
        random_state=42
    )

# ----------------------- METRICS ----------------------
def regression_metrics(y_true, y_pred):
    mse = mean_squared_error(y_true, y_pred)   # older sklearn has no `squared=`
    rmse = float(np.sqrt(mse))
    return {
        "MAE":  mean_absolute_error(y_true, y_pred),
        "RMSE": rmse,
        "R2":   r2_score(y_true, y_pred)
    }

def classification_metrics(y_true, y_pred_cont, thr=0.5):
    # y_true_class from continuous stress target:
    y_true_cls = (y_true >= 0.5).astype(int)
    y_pred_cls = (y_pred_cont >= thr).astype(int)

    out = {
        "ACC": accuracy_score(y_true_cls, y_pred_cls),
        "F1":  f1_score(y_true_cls, y_pred_cls, zero_division=0),
        "ROC_AUC": roc_auc_score(y_true_cls, y_pred_cont),
        "PR_AUC":  average_precision_score(y_true_cls, y_pred_cont),
        "THR": thr
    }
    return out

def best_f1_threshold(y_true, y_pred_cont):
    y_true_cls = (y_true >= 0.5).astype(int)
    # sweep thresholds from 0.1..0.9
    thrs = np.linspace(0.1, 0.9, 81)
    best_thr, best_f1 = 0.5, -1
    for t in thrs:
        f1 = f1_score(y_true_cls, (y_pred_cont >= t).astype(int), zero_division=0)
        if f1 > best_f1:
            best_f1, best_thr = f1, t
    return best_thr, best_f1

# ----------------------- PLOTS ------------------------
def plot_timeseries(t1, y_true, y_pred, title, path):
    plt.figure(figsize=(10,4))
    plt.plot(t1, y_true, label="True", linewidth=1.5)
    plt.plot(t1, y_pred, label="Pred", linewidth=1.2)
    plt.ylim(-0.05, 1.05)
    plt.title(title)
    plt.xlabel("t1_unix (s)")
    plt.ylabel("stress_prob")
    plt.legend()
    plt.tight_layout()
    plt.savefig(path)
    plt.close()

def plot_scatter(y_true, y_pred, title, path):
    plt.figure(figsize=(5,5))
    plt.scatter(y_true, y_pred, s=16, alpha=0.7)
    lims = [min(0, y_true.min(), y_pred.min()), max(1, y_true.max(), y_pred.max())]
    plt.plot(lims, lims, 'k--', linewidth=1)
    plt.xlim(lims); plt.ylim(lims)
    plt.xlabel("True"); plt.ylabel("Predicted")
    plt.title(title)
    plt.tight_layout()
    plt.savefig(path)
    plt.close()

def plot_residual_hist(y_true, y_pred, title, path):
    resid = y_pred - y_true
    plt.figure(figsize=(6,4))
    plt.hist(resid, bins=30)
    plt.title(title + " (Residuals)")
    plt.xlabel("Pred - True"); plt.ylabel("Count")
    plt.tight_layout()
    plt.savefig(path)
    plt.close()

def plot_importances(model, feat_names, title, path, top_k=20):
    if not hasattr(model, "feature_importances_"):
        return
    imp = np.asarray(model.feature_importances_)
    order = np.argsort(imp)[::-1][:top_k]
    plt.figure(figsize=(8, max(3, int(top_k*0.4))))
    plt.barh(np.array(feat_names)[order][::-1], imp[order][::-1])
    plt.title(title + " (Top importances)")
    plt.tight_layout()
    plt.savefig(path)
    plt.close()

# ----------------------- RUN -------------------------
def main():
    df = load_clean_dataframe(CSV_PATH)
    print(f"Loaded {len(df)} rows after filtering.")

    emb_cols, mvp_cols, mask_cols, y_col = pick_columns(df)
    y = df[y_col].astype(float)
    t1 = df["t1_unix"] if "t1_unix" in df.columns else pd.Series(np.arange(len(df)))

    # A) Embeddings-only
    X_emb = df[emb_cols].copy()
    Xtr_e, Xte_e, ytr_e, yte_e = chrono_train_test_split(X_emb, y, test_frac=0.2)
    model_emb = make_gbrt(n_estimators=600)
    model_emb.fit(Xtr_e, ytr_e)
    check_is_fitted(model_emb)
    pred_tr_e = model_emb.predict(Xtr_e)
    pred_te_e = model_emb.predict(Xte_e)

    # B) Embeddings + MVP (+ masks)
    hybrid_cols = emb_cols + mvp_cols + mask_cols
    X_hyb = df[hybrid_cols].copy()
    Xtr_h, Xte_h, ytr_h, yte_h = chrono_train_test_split(X_hyb, y, test_frac=0.2)
    model_hyb = make_gbrt(n_estimators=800)
    model_hyb.fit(Xtr_h, ytr_h)
    check_is_fitted(model_hyb)
    pred_tr_h = model_hyb.predict(Xtr_h)
    pred_te_h = model_hyb.predict(Xte_h)

    # ---------------- Metrics ----------------
    reg_emb_tr = regression_metrics(ytr_e, pred_tr_e)
    reg_emb_te = regression_metrics(yte_e, pred_te_e)

    reg_hyb_tr = regression_metrics(ytr_h, pred_tr_h)
    reg_hyb_te = regression_metrics(yte_h, pred_te_h)

    # Classification-style metrics (default thr=0.5)
    cls_emb_05 = classification_metrics(yte_e, pred_te_e, thr=0.5)
    cls_hyb_05 = classification_metrics(yte_h, pred_te_h, thr=0.5)

    # Also compute "best F1 threshold" on TRAIN, then evaluate that on TEST
    thr_e, f1e = best_f1_threshold(ytr_e, pred_tr_e)
    thr_h, f1h = best_f1_threshold(ytr_h, pred_tr_h)
    cls_emb_best = classification_metrics(yte_e, pred_te_e, thr=thr_e)
    cls_hyb_best = classification_metrics(yte_h, pred_te_h, thr=thr_h)

    # Print table
    def row(label, reg_te, cls_05, cls_best):
        return {
            "Model": label,
            "MAE": reg_te["MAE"],
            "RMSE": reg_te["RMSE"],
            "R2": reg_te["R2"],
            "ACC@0.5": cls_05["ACC"],
            "F1@0.5":  cls_05["F1"],
            "ROC_AUC": cls_05["ROC_AUC"],
            "PR_AUC":  cls_05["PR_AUC"],
            "ACC@best": classification_metrics(yte_e if "Emb" in label else yte_h,
                                               pred_te_e if "Emb" in label else pred_te_h,
                                               thr=cls_emb_best["THR"] if "Emb" in label else cls_hyb_best["THR"])["ACC"],
            "F1@best":  cls_best["F1"],
            "THR*":     cls_best["THR"]
        }

    summary = pd.DataFrame([
        row("Embeddings-only", reg_emb_te, cls_emb_05, cls_emb_best),
        row("Embeddings+MVP",  reg_hyb_te, cls_hyb_05, cls_hyb_best),
    ])
    print("\n=== TEST METRICS (chronological split) ===")
    print(summary.to_string(index=False))

    # ---------------- Plots ----------------
    # Indices for test part (chronological)
    t1_te_e = t1.iloc[len(Xtr_e):]
    t1_te_h = t1.iloc[len(Xtr_h):]

    plot_timeseries(t1_te_e, yte_e, pred_te_e,
                    "Embeddings-only: True vs Pred (Test)",
                    os.path.join(FIG_DIR, "emb_ts.png"))
    plot_scatter(yte_e, pred_te_e,
                 "Embeddings-only: Scatter (Test)",
                 os.path.join(FIG_DIR, "emb_scatter.png"))
    plot_residual_hist(yte_e, pred_te_e,
                       "Embeddings-only",
                       os.path.join(FIG_DIR, "emb_resid.png"))
    plot_importances(model_emb, X_emb.columns,
                     "Embeddings-only", os.path.join(FIG_DIR, "emb_importance.png"))

    plot_timeseries(t1_te_h, yte_h, pred_te_h,
                    "Embeddings+MVP: True vs Pred (Test)",
                    os.path.join(FIG_DIR, "hyb_ts.png"))
    plot_scatter(yte_h, pred_te_h,
                 "Embeddings+MVP: Scatter (Test)",
                 os.path.join(FIG_DIR, "hyb_scatter.png"))
    plot_residual_hist(yte_h, pred_te_h,
                       "Embeddings+MVP",
                       os.path.join(FIG_DIR, "hyb_resid.png"))
    plot_importances(model_hyb, X_hyb.columns,
                     "Embeddings+MVP", os.path.join(FIG_DIR, "hyb_importance.png"))

    # ---------------- Save models & meta ----------------
    dump(model_emb, os.path.join(OUT_DIR, "personal_head_emb.joblib"))
    dump(model_hyb, os.path.join(OUT_DIR, "personal_head_hybrid.joblib"))

    meta = {
        "csv_path": CSV_PATH,
        "target": "stress_prob",
        "emb_cols": list(X_emb.columns),
        "mvp_cols": MVP_FIELDS,
        "mask_cols": [c for c in ["has_mouse_emb","has_keys_emb"] if c in X_hyb.columns],
        "hybrid_cols": list(X_hyb.columns),
        "thresholds": {
            "emb_best_f1_thr": float(thr_e),
            "hyb_best_f1_thr": float(thr_h)
        }
    }
    with open(os.path.join(OUT_DIR, "personal_head_meta.json"), "w") as f:
        json.dump(meta, f, indent=2)

    print(f"\nSaved models & plots to: {OUT_DIR}")
    print(f"Figures: {FIG_DIR}")
    print("\n*THR is selected to maximize F1 on the TRAIN split; metrics shown at both THR=0.5 and THR=THR*.")

if __name__ == "__main__":
    main()
