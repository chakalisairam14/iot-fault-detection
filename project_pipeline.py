# project_pipeline.py
# IoT-Based Machine Fault Detection - End-to-End Pipeline
# Author: <you>
# Usage:
#   python project_pipeline.py --data_dir data --out_dir outputs --fds FD001 FD002 FD003 FD004

import os
import json
import argparse
from pathlib import Path
import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd

# ML / Metrics
from sklearn.preprocessing import MinMaxScaler
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import GridSearchCV, StratifiedKFold
from sklearn.metrics import (
    roc_auc_score, roc_curve, precision_recall_curve, average_precision_score,
    classification_report, confusion_matrix, precision_recall_fscore_support
)
from sklearn.utils.validation import check_is_fitted

# Viz
import matplotlib.pyplot as plt
import seaborn as sns

# Persistence
import joblib

# PDF report
from reportlab.lib.pagesizes import A4
from reportlab.pdfgen import canvas
from reportlab.lib.utils import ImageReader


model = LogisticRegression(max_iter=5000, solver='lbfgs')  # or solver='saga' for large data

# -----------------------
# Helpers
# -----------------------

data_dir = Path(r"C:\Users\chaka\OneDrive\Desktop\projectfailure\Data")
COLUMNS = ["unit_id","time_cycles"] + [f"op_setting_{i}" for i in range(1,4)] + [f"sensor_{i}" for i in range(1,22)]

def load_fd_split(fd: str, data_dir: Path):
    train_path = data_dir / f"train_{fd}.txt"
    test_path  = data_dir / f"test_{fd}.txt"
    rul_path   = data_dir / f"RUL_{fd}.txt"
    if not train_path.exists(): raise FileNotFoundError(train_path)
    if not test_path.exists():  raise FileNotFoundError(test_path)
    if not rul_path.exists():   raise FileNotFoundError(rul_path)
    train = pd.read_csv(train_path, sep=r"\s+", header=None, names=COLUMNS, usecols=range(26))
    test  = pd.read_csv(test_path,  sep=r"\s+", header=None, names=COLUMNS, usecols=range(26))
    rul   = pd.read_csv(rul_path,   header=None, names=["RUL"])
    return train, test, rul

def add_labels_and_merge_test_rul(train, test, rul, horizon):
    train = train.copy()
    train["RUL"] = train.groupby("unit_id")["time_cycles"].transform(lambda x: x.max() - x)
    train["failure_24h"] = (train["RUL"] <= horizon).astype(int)

    last_cycles = test.groupby("unit_id")["time_cycles"].max().reset_index()
    last_cycles["RUL"] = rul["RUL"].values
    test = test.merge(last_cycles, on="unit_id", how="left")
    
    # Fix duplicate time_cycles column
    if "time_cycles_x" in test.columns:
        test = test.rename(columns={"time_cycles_x": "time_cycles"})
    elif "time_cycles_y" in test.columns:
        test = test.rename(columns={"time_cycles_y": "time_cycles"})

    test["failure_24h"] = (test["RUL"] <= horizon).astype(int)
    return train, test

def robust_clip(df, cols, k=5.0):
    """Clip extreme anomalies using median±k*IQR (helps with spikes)."""
    df = df.copy()
    for c in cols:
        q1, q3 = df[c].quantile([0.25, 0.75])
        iqr = q3 - q1
        low, high = q1 - k*iqr, q3 + k*iqr
        df[c] = df[c].clip(lower=low, upper=high)
    return df

def feature_engineering(df):
    df = df.copy()
    
    # Rolling stats + rate-of-change for sensors
    for s in [4, 7, 12]:
        col = f"sensor_{s}"
        if col in df.columns:
            df[f"roll_mean_{col}"] = df.groupby("unit_id")[col].transform(lambda x: x.rolling(5, min_periods=1).mean())
            df[f"roll_std_{col}"]  = df.groupby("unit_id")[col].transform(lambda x: x.rolling(5, min_periods=1).std())
            df[f"{col}_roc"]       = df.groupby("unit_id")[col].diff()
            df[f"{col}_roc_s5"]    = df.groupby("unit_id")[f"{col}_roc"].transform(lambda x: x.rolling(5, min_periods=1).mean())
    
    # Interaction & ratio features with ops
    for op in [1, 2, 3]:
        op_col = f"op_setting_{op}"
        if op_col in df.columns:
            for s in [7, 11]:
                sensor_col = f"sensor_{s}"
                if sensor_col in df.columns:
                    df[f"op{op}_x_sensor{s}"] = df[op_col] * df[sensor_col]
                    df[f"sensor{s}_over_op{op}"] = df[sensor_col] / df[op_col].replace(0, np.nan)
    
    # Aging feature
    df["cycles_since_start"] = df.groupby("unit_id").cumcount()
    
    return df.fillna(0.0)


def scale_columns(train, test, cols):
    scaler = MinMaxScaler()
    train[cols] = scaler.fit_transform(train[cols])
    test[cols]  = scaler.transform(test[cols])
    return train, test, scaler

def tune_models(X, y, seed=42):
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=seed)

    lr_grid = {"C":[0.1,0.5,1,2,5,10], "penalty":["l2"], "solver":["lbfgs","liblinear"]}
    lr_gs = GridSearchCV(LogisticRegression(max_iter=2000), lr_grid, scoring="roc_auc", cv=cv, n_jobs=-1)
    lr_gs.fit(X, y)

    dt_grid = {"max_depth":[6,8,10,12,15], "min_samples_split":[2,5,10], "min_samples_leaf":[1,2,5]}
    dt_gs = GridSearchCV(DecisionTreeClassifier(random_state=seed), dt_grid, scoring="roc_auc", cv=cv, n_jobs=-1)
    dt_gs.fit(X, y)

    return lr_gs, dt_gs

def pick_threshold(y_true, proba, goal="f1", min_precision=None):
    """Find a decision threshold:
       - goal='f1' (maximize F1)
       - or ensure precision≥min_precision with best recall."""
    precisions, recalls, thresholds = precision_recall_curve(y_true, proba)
    thresholds = np.r_[0, thresholds]  # align with precisions/recalls length
    if min_precision is not None:
        mask = precisions >= min_precision
        if mask.any():
            # among those meeting precision, maximize recall
            idx = np.argmax(recalls[mask])
            return thresholds[mask][idx]
    # default: maximize F1
    f1s = (2*precisions*recalls) / (precisions+recalls + 1e-9)
    idx = np.argmax(f1s)
    return thresholds[idx]

def business_case(y_true, y_pred, cost_fail=10000, cost_pm=1000):
    """Simple business model:
    - If we predict positive, we do preventive maintenance costing cost_pm.
    - If actual failure occurs and we didn't predict, cost_fail occurs.
    Returns estimated total cost and savings vs. 'do nothing'."""
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)

    # cost if we do nothing: every failure costs cost_fail
    baseline_cost = (y_true == 1).sum() * cost_fail

    # our policy:
    # PM cost for every predicted positive
    pm_cost = (y_pred == 1).sum() * cost_pm
    # Failure cost for false negatives (missed failures)
    missed_fail_cost = ((y_true == 1) & (y_pred == 0)).sum() * cost_fail
    total_cost = pm_cost + missed_fail_cost

    savings = baseline_cost - total_cost
    return dict(baseline_cost=int(baseline_cost), total_cost=int(total_cost), savings=int(savings))

def plot_and_save_roc(y_true, proba, title, out_png):
    fpr, tpr, _ = roc_curve(y_true, proba)
    auc = roc_auc_score(y_true, proba)
    plt.figure()
    plt.plot(fpr, tpr, label=f"AUC={auc:.3f}")
    plt.plot([0,1],[0,1],'--',color='gray')
    plt.xlabel("False Positive Rate"); plt.ylabel("True Positive Rate")
    plt.title(title); plt.legend()
    plt.tight_layout()
    plt.savefig(out_png, dpi=160); plt.close()
    return auc

def plot_and_save_pr(y_true, proba, title, out_png):
    precision, recall, _ = precision_recall_curve(y_true, proba)
    ap = average_precision_score(y_true, proba)
    plt.figure()
    plt.plot(recall, precision, label=f"AP={ap:.3f}")
    plt.xlabel("Recall"); plt.ylabel("Precision")
    plt.title(title); plt.legend()
    plt.tight_layout()
    plt.savefig(out_png, dpi=160); plt.close()
    return ap

def plot_and_save_cm(y_true, y_pred, title, out_png):
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(4,3))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues")
    plt.title(title); plt.xlabel("Predicted"); plt.ylabel("True")
    plt.tight_layout()
    plt.savefig(out_png, dpi=160); plt.close()

def make_pdf_report(pdf_path, meta, images):
    pdf_path = Path(pdf_path)
    pdf_path.parent.mkdir(parents=True, exist_ok=True)
    c = canvas.Canvas(str(pdf_path), pagesize=A4)
    W, H = A4

    # Title page
    c.setFont("Helvetica-Bold", 20)
    c.drawString(72, H-100, "IoT-Based Machine Fault Detection")
    c.setFont("Helvetica", 12)
    c.drawString(72, H-130, "Objective: Predict machine failure 24 hours in advance using sensor readings.")
    c.drawString(72, H-150, f"Datasets: NASA C-MAPSS ({', '.join(meta['fds'])})")
    c.drawString(72, H-170, f"Best Model: {meta['best_model']} | ROC-AUC: {meta['best_auc']:.3f}")
    c.drawString(72, H-190, f"Threshold: {meta['threshold']:.3f}")
    c.showPage()

    # Methods
    c.setFont("Helvetica-Bold", 16)
    c.drawString(72, H-72, "Methods")
    c.setFont("Helvetica", 11)
    text = c.beginText(72, H-100)
    text.textLines(
        "• Preprocessing: anomaly clipping (IQR), per-FD label generation (RUL→binary <=24), "
        "per-train scaling (MinMax)."
        "\n• Feature Engineering: rolling mean/std (window=5), rate-of-change and smoothed ROC, "
        "op-setting interactions, ratios, cycles-since-start."
        "\n• Models: Logistic Regression, Decision Tree with GridSearchCV (5-fold Stratified)."
        "\n• Metrics: ROC-AUC, PR-AUC, Confusion Matrix; threshold chosen to maximize F1 (or set precision floor)."
    )
    c.drawText(text)
    c.showPage()

    # Results page with images
    c.setFont("Helvetica-Bold", 16); c.drawString(72, H-72, "Results")
    y = H-100
    for label, img_path in images.items():
        if not Path(img_path).exists(): continue
        c.setFont("Helvetica", 11); c.drawString(72, y, label); y -= 12
        c.drawImage(ImageReader(img_path), 72, y-220, width=400, height=220, preserveAspectRatio=True, anchor='sw')
        y -= 240
        if y < 150:
            c.showPage(); y = H-100

    # Business case
    c.showPage()
    c.setFont("Helvetica-Bold", 16); c.drawString(72, H-72, "Business Case")
    c.setFont("Helvetica", 11)
    bc = meta["business"]
    text = c.beginText(72, H-100)
    text.textLines(
        f"Assumptions: Cost of failure = ${bc['assumptions']['cost_fail']:,}, "
        f"Preventive maintenance = ${bc['assumptions']['cost_pm']:,}."
        f"\nBaseline (do nothing) cost: ${bc['baseline_cost']:,}"
        f"\nStrategy total cost:       ${bc['total_cost']:,}"
        f"\nEstimated savings:         ${bc['savings']:,}"
    )
    c.drawText(text)

    # Conclusion
    c.showPage()
    c.setFont("Helvetica-Bold", 16); c.drawString(72, H-72, "Conclusion")
    c.setFont("Helvetica", 11)
    text = c.beginText(72, H-100)
    text.textLines(
        f"• The selected model ({meta['best_model']}) achieves ROC-AUC {meta['best_auc']:.3f} with an "
        f"operational threshold of {meta['threshold']:.2f} optimized for early high-risk detection.\n"
        "• Rolling and interaction features capture degradation patterns; dashboard supports per-machine risk tracking.\n"
        "• Next steps: add LSTM/GRU sequence model baseline, calibrate probabilities (Platt/Isotonic), and deploy "
        "the scoring service behind the dashboard."
    )
    c.drawText(text)
    c.showPage(); c.save()

# -----------------------
# Main pipeline
# -----------------------
def main(args):
    data_dir = Path(args.data_dir)
    out_dir  = Path(args.out_dir)
    proc_dir = out_dir / "processed_data"
    art_dir  = out_dir / "artifacts"
    fig_dir  = out_dir / "figures"
    report_dir = out_dir / "reports"
    for d in [proc_dir, art_dir, fig_dir, report_dir]:
        d.mkdir(parents=True, exist_ok=True)

    fds = args.fds or ["FD001","FD002","FD003","FD004"]
    horizon = args.horizon

    all_tr, all_te = [], []

    for fd in fds:
        tr, te, rul = load_fd_split(fd, data_dir)

        # anomaly clipping (only sensors)
        sensor_cols = [c for c in tr.columns if c.startswith("sensor_")]
        tr = robust_clip(tr, sensor_cols)
        te = robust_clip(te, sensor_cols)

        tr, te = add_labels_and_merge_test_rul(tr, te, rul, horizon)

        # features
        # ------------------------

        # 1. Feature engineering already done
        tr = feature_engineering(tr)
        te = feature_engineering(te)
# 1. Define features to scale (all except IDs, labels, fd)
# ------------------------
        drop_cols = {"unit_id","time_cycles","RUL","failure_24h"}
        feature_cols_to_scale = [c for c in tr.columns if c not in drop_cols]

# ------------------------
# 2. Fit scaler on train, transform train and test
# ------------------------
        scaler = MinMaxScaler()
        tr[feature_cols_to_scale] = scaler.fit_transform(tr[feature_cols_to_scale])
        te[feature_cols_to_scale] = scaler.transform(te[feature_cols_to_scale])

# ------------------------
# 3. Save per-FD scaler if needed
# ------------------------
        joblib.dump(scaler, art_dir / f"scaler_{fd}.pkl")
  


        # scale sensor columns only (engineered can remain raw; optional to scale)
        tr, te, scaler = scale_columns(tr, te, [c for c in tr.columns if c.startswith("sensor_")])

        tr["fd"] = fd; te["fd"] = fd
        tr.to_parquet(proc_dir / f"train_{fd}.parquet", index=False)
        te.to_parquet(proc_dir / f"test_{fd}.parquet",  index=False)
        all_tr.append(tr); all_te.append(te)
        print(f"✅ {fd} processed: train={tr.shape}, test={te.shape}")

    train_df = pd.concat(all_tr, ignore_index=True)
    test_df  = pd.concat(all_te, ignore_index=True)
    train_df.to_parquet(proc_dir / "combined_train.parquet", index=False)
    test_df.to_parquet(proc_dir / "combined_test.parquet", index=False)
    print("✅ Combined:", train_df.shape, test_df.shape)

    # features/labels
    drop_cols = {"unit_id","time_cycles","RUL","failure_24h","fd"}
    feature_cols = [c for c in train_df.columns if c not in drop_cols]
    X_train, y_train = train_df[feature_cols], train_df["failure_24h"]
    X_test,  y_test  = test_df[feature_cols],  test_df["failure_24h"]

    # tune
    lr_gs, dt_gs = tune_models(X_train, y_train, seed=args.seed)
    lr_best, dt_best = lr_gs.best_estimator_, dt_gs.best_estimator_

    # evaluate
    lr_auc = roc_auc_score(y_test, lr_best.predict_proba(X_test)[:,1])
    dt_auc = roc_auc_score(y_test, dt_best.predict_proba(X_test)[:,1])

    if lr_auc >= dt_auc:
        best_model, best_name, best_auc = lr_best, "Logistic Regression", lr_auc
    else:
        best_model, best_name, best_auc = dt_best, "Decision Tree", dt_auc

    # threshold optimization (maximize F1, or enforce precision floor if provided)
    proba = best_model.predict_proba(X_test)[:,1]
    thr = pick_threshold(y_test, proba, goal="f1",
                         min_precision=args.min_precision if args.min_precision else None)
    y_pred = (proba >= thr).astype(int)

    # metrics + plots
    cm_png  = fig_dir / "confusion_matrix.png"
    roc_png = fig_dir / "roc_curve.png"
    pr_png  = fig_dir / "pr_curve.png"

    plot_and_save_cm(y_test, y_pred, f"{best_name} @thr={thr:.2f}", cm_png)
    auc_val = plot_and_save_roc(y_test, proba, f"{best_name} ROC", roc_png)
    ap_val  = plot_and_save_pr(y_test, proba, f"{best_name} PR",  pr_png)

    cls_report = classification_report(y_test, y_pred, output_dict=True)
    with open(art_dir / "metrics.json","w") as f:
        json.dump({
            "best_model": best_name,
            "roc_auc": auc_val,
            "pr_auc": ap_val,
            "threshold": float(thr),
            "classification_report": cls_report
        }, f, indent=2)

    # business case
    bc = business_case(y_test, y_pred, cost_fail=args.cost_fail, cost_pm=args.cost_pm)
    bc["assumptions"] = {"cost_fail": args.cost_fail, "cost_pm": args.cost_pm}
    with open(art_dir / "business_case.json","w") as f:
        json.dump(bc, f, indent=2)

    # persist artifacts
    joblib.dump(best_model, art_dir / "model.pkl")
    # fit a scaler on all sensors across train for dashboard scoring consistency
    feature_cols = [c for c in train_df.columns if c not in {"unit_id","time_cycles","RUL","failure_24h","fd"}]
    scaler_all = MinMaxScaler().fit(train_df[feature_cols])
    joblib.dump(scaler_all, art_dir / "scaler.pkl")

    with open(art_dir / "feature_names.json","w") as f:
        json.dump(feature_cols, f)

    # save a ready-to-use test set for the dashboard
    test_df.to_parquet(proc_dir / "combined_test.parquet", index=False)

    # PDF report
    make_pdf_report(
        pdf_path=report_dir / "project_report.pdf",
        meta={
            "fds": fds,
            "best_model": best_name,
            "best_auc": auc_val,
            "threshold": float(thr),
            "business": bc
        },
        images={
            "ROC Curve": roc_png,
            "Precision-Recall Curve": pr_png,
            "Confusion Matrix": cm_png
        }
    )

    print("✅ Done")
    print(f"Best model: {best_name} | ROC-AUC={best_auc:.3f} | Threshold={thr:.2f}")
    print(f"Artifacts: {art_dir.resolve()}")
    print(f"Figures:   {fig_dir.resolve()}")
    print(f"Report:    {(report_dir / 'project_report.pdf').resolve()}")

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_dir", type=str, default="data", help="Folder with NASA C-MAPSS txt files")
    ap.add_argument("--out_dir", type=str, default="outputs", help="Folder to write processed data/artifacts/report")
    ap.add_argument("--fds", nargs="*", default=["FD001","FD002","FD003","FD004"], help="Which FD sets to use")
    ap.add_argument("--horizon", type=int, default=24, help="Failure horizon in cycles (24)")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--min_precision", type=float, default=None, help="If set, choose threshold with precision≥value")
    ap.add_argument("--cost_fail", type=int, default=10000, help="Cost of an unplanned failure")
    ap.add_argument("--cost_pm",   type=int, default=1000, help="Cost of a preventive maintenance action")
    args = ap.parse_args()
    main(args)
