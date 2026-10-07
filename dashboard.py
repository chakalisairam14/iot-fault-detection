# dashboard.py
# Run: streamlit run dashboard.py

import json
import time
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

import dash
from dash import html, Input, Output, State, dcc
import dash_bootstrap_components as dbc

# ----------------------------
# Paths / Artifacts
# ----------------------------
OUT_DIR   = Path("outputs")
ART_DIR   = OUT_DIR / "artifacts"
PROC_DIR  = OUT_DIR / "processed_data"
REPORTS   = OUT_DIR / "reports"

MODEL_PATH  = ART_DIR / "model.pkl"
SCALER_PATH = ART_DIR / "scaler.pkl"
FEATS_PATH  = ART_DIR / "feature_names.json"
TEST_PATH   = PROC_DIR / "combined_test.parquet"
PDF_PATH    = REPORTS / "project_report.pdf"

# ----------------------------
# Page config
# ----------------------------
st.set_page_config(page_title="IoT Fault Detection", page_icon="🛠️", layout="wide")

# ----------------------------
# Global styles (dark / light friendly)
# ----------------------------
def inject_css():
    st.markdown(
        """
        <style>
        /* Base */
        .main { padding-top: 0rem; }
        .block-container { padding-top: 2.5rem; padding-bottom: 2rem; }
        /* Header bar */
        .app-header { 
            display:flex; align-items:center; justify-content:space-between;
            padding: 12px 16px; border-radius: 12px; 
            background: linear-gradient(135deg, rgba(99,102,241,.15), rgba(236,72,153,.10));
            margin-bottom: 12px;
            position: -webkit-sticky;
        position: sticky;
        top: 0;
        z-index: 999;
        }
        .app-title { font-size: 1.25rem; font-weight: 700; margin: 0; }
        .app-sub { opacity:.7; font-size:.9rem; margin-top:2px }
        .kpi-card {
            border-radius:16px; padding:16px; border:1px solid rgba(128,128,128,.2);
            background: rgba(255,255,255,.55);
        }
        [data-baseweb="select"] { font-size: 0.95rem; }
        .metric { text-align:center; }
        .footnote { opacity:.7; font-size:.85rem; }
        </style>
        """,
        unsafe_allow_html=True,
    )

inject_css()

# ----------------------------
# Cache / Loading
# ----------------------------
@st.cache_resource(show_spinner=True)
def load_assets():
    model = joblib.load(MODEL_PATH)
    scaler = joblib.load(SCALER_PATH)
    feats  = json.loads(FEATS_PATH.read_text())

    df = pd.read_parquet(TEST_PATH)
    # drop duplicate columns if any
    df = df.loc[:, ~df.columns.duplicated()]

    # Standardize time_cycles name if accidentally suffixed by merges
    if "time_cycles_x" in df.columns and "time_cycles" not in df.columns:
        df = df.rename(columns={"time_cycles_x": "time_cycles"})
    if "time_cycles_y" in df.columns and "time_cycles" not in df.columns:
        df = df.rename(columns={"time_cycles_y": "time_cycles"})

    # Ensure required cols
    for c in ["unit_id", "time_cycles", "fd"]:
        if c not in df.columns:
            raise ValueError(f"Required column '{c}' missing from dataset: {TEST_PATH}")

    return model, scaler, feats, df

# Load once
try:
    model, scaler, FEATURE_COLS, raw_df = load_assets()
except Exception as e:
    st.error(f"❌ Failed to load assets: {e}")
    st.stop()

# ----------------------------
# Header (Navbar-like)
# ----------------------------
colL, colR = st.columns([4, 1])
with colL:
    st.markdown(
        """
        <div class="app-header">
            <div>
                <div class="app-title">📊 IoT Machine Fault Detection</div>
                <div class="app-sub">Live risk scoring, sensor analytics, and explainability</div>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )
with colR:
    # Dark mode toggle (visual only; Streamlit theme is set in settings.toml if needed)
    dark = st.toggle("🌙 Dark mode", value=False, help="UI preference (charts auto-adapt)")

# ----------------------------
# Sidebar: Filters + Controls
# ----------------------------
st.sidebar.header("Filters")

# Auto refresh
refresh_rate = st.sidebar.slider("Auto-refresh (sec)", 0, 30, 0, help="Simulate live updates")
if refresh_rate > 0:
    time.sleep(refresh_rate)
    st.rerun()

# FD subset
df = raw_df.copy()
fd_options = ["All"] + sorted(df["fd"].unique().tolist())
fd_sel = st.sidebar.selectbox("FD Subset", fd_options, index=0, key="fd_sel")
if fd_sel != "All":
    df = df[df["fd"] == fd_sel].copy()

# Unit select
units = sorted(df["unit_id"].unique())
if len(units) == 0:
    st.warning("No units found for the current filter.")
    st.stop()

unit_id = st.sidebar.selectbox("Machine ID", units, key="unit_sel")

# Threshold control
thr_default = 0.5
thr = st.sidebar.slider("Alert Threshold", 0.0, 1.0, float(thr_default), 0.01, help="Classify failure if risk ≥ threshold")

# Sensor pick
sensor_cols_all = [c for c in df.columns if c.startswith("sensor_")]
default_sensors = [s for s in ["sensor_4", "sensor_7", "sensor_12"] if s in sensor_cols_all]
sensor_pick = st.sidebar.multiselect("Sensors to plot", sensor_cols_all, default=default_sensors, key="sensor_pick")

# Time window
max_cycle = int(df["time_cycles"].max())
window = st.sidebar.slider("Show last N cycles", 50, max(50, max_cycle), min(500, max_cycle), step=50)

# ----------------------------
# Prepare unit data + predictions
# ----------------------------
unit_df_all = df[df["unit_id"] == unit_id].sort_values("time_cycles").copy()
if unit_df_all.empty:
    st.info("No rows for the selected unit.")
    st.stop()

# Prediction (use saved feature list)
missing = [c for c in FEATURE_COLS if c not in unit_df_all.columns]
if missing:
    st.error(f"Model features missing in dataset: {missing[:10]}{'...' if len(missing)>10 else ''}")
    st.stop()

# Compute risk
proba = model.predict_proba(unit_df_all[FEATURE_COLS])[:, 1]
unit_df_all = unit_df_all.assign(risk_score=proba, pred=(proba >= thr).astype(int))

# Apply time window
cut = unit_df_all["time_cycles"].max() - window
unit_df = unit_df_all[unit_df_all["time_cycles"] >= cut].copy()

latest = unit_df_all.iloc[-1]

# ----------------------------
# Alerts
# ----------------------------
if latest["risk_score"] >= max(thr, 0.8):
    st.warning("🚨 HIGH FAILURE RISK! Immediate maintenance recommended!", icon="⛑️")
elif latest["risk_score"] >= max(thr, 0.5):
    st.info("⚠️ Warning: failure risk increasing. Monitor closely.", icon="⚠️")

# ----------------------------
# KPI row
# ----------------------------
k1, k2, k3, k4 = st.columns(4)
k1.metric("Machine ID", int(unit_id))
k2.metric("Current Risk", f"{latest['risk_score']:.3f}")
k3.metric(f"Predicted Failure (≥{thr:.2f})", "Yes" if latest["pred"] == 1 else "No")
k4.metric("Last Cycle", int(latest["time_cycles"]))

# ----------------------------
# Tabs: Overview / Sensors / Analytics / Explainability / Data
# ----------------------------
tab_overview, tab_sensors, tab_analytics, tab_explain, tab_data = st.tabs(
    ["Overview", "Sensors", "Analytics", "Explainability", "Data"]
)

# ----- Overview
with tab_overview:
    c1, c2 = st.columns([1, 1])

    # Gauge
    with c1:
        st.subheader("Failure Probability Gauge")
        fig_g = go.Figure(
            go.Indicator(
                mode="gauge+number",
                value=float(latest["risk_score"] * 100),
                title={'text': "Failure Probability (%)"},
                gauge={
                    'axis': {'range': [0, 100]},
                    'bar': {'color': "crimson"},
                    'steps': [
                        {'range': [0, 50],  'color': "lightgreen"},
                        {'range': [50, 80], 'color': "gold"},
                        {'range': [80, 100],'color': "salmon"}
                    ]
                }
            )
        )
        st.plotly_chart(fig_g, use_container_width=True)

    # Risk over time
    with c2:
        st.subheader("Risk Score Over Time")
        fig_risk = px.line(
            unit_df, x="time_cycles", y="risk_score",
            markers=True, title="Predicted Failure Risk", labels={"time_cycles":"Cycle","risk_score":"Risk"}
        )
        st.plotly_chart(fig_risk, use_container_width=True)

    st.markdown("---")
    c3, c4 = st.columns([1, 1])

    with c3:
        st.subheader("Prediction vs Threshold")
        fig_thr = px.area(
            unit_df, x="time_cycles", y="risk_score",
            title="Risk and Decision Threshold", labels={"time_cycles":"Cycle","risk_score":"Risk"}
        )
        fig_thr.add_hline(y=thr, line_dash="dash", line_color="gray", annotation_text=f"thr={thr:.2f}")
        st.plotly_chart(fig_thr, use_container_width=True)

    with c4:
        st.subheader("Actual Failures (if labeled)")
        if "failure_24h" in unit_df.columns:
            fig_sc = px.scatter(
                unit_df, x="time_cycles", y="risk_score",
                color=unit_df["failure_24h"].astype(str),
                title="Risk vs Actual Label", labels={"color":"Actual Failure"}
            )
            st.plotly_chart(fig_sc, use_container_width=True)
        else:
            st.info("No ground-truth labels found for this subset.")

# ----- Sensors
with tab_sensors:
    st.subheader("Sensor Trends")
    if sensor_pick:
        long_df = unit_df[["time_cycles"] + sensor_pick].melt(
            id_vars="time_cycles", var_name="sensor", value_name="value"
        )
        fig_sensors = px.line(
            long_df, x="time_cycles", y="value", color="sensor",
            title="Selected Sensor Readings", labels={"time_cycles":"Cycle","value":"Reading"}
        )
        st.plotly_chart(fig_sensors, use_container_width=True)
    else:
        st.info("Pick one or more sensors in the sidebar.")

    st.markdown("#### Correlation (current window)")
    sens_for_corr = [c for c in sensor_cols_all if c in unit_df.columns]
    if len(sens_for_corr) >= 2:
        corr = unit_df[sens_for_corr].corr()
        fig_corr = px.imshow(corr, text_auto=False, aspect="auto", title="Sensor Correlation Heatmap")
        st.plotly_chart(fig_corr, use_container_width=True)
    else:
        st.info("Not enough sensors to compute correlation.")

# ----- Analytics
with tab_analytics:
    st.subheader("Fleet Snapshot (Filtered)")
    # Risk distribution
    risk_hist = px.histogram(
        df.assign(risk_score=model.predict_proba(df[FEATURE_COLS])[:,1]),
        x="risk_score", nbins=40, title="Risk Distribution (All units – filtered)"
    )
    st.plotly_chart(risk_hist, use_container_width=True)
# ----- Analytics
with tab_analytics:
    # --- Section 1: Enhanced EDA ---
    st.subheader("📊 Exploratory Data Analysis (EDA) - Filtered Fleet")
    
    # Calculate risk for the entire filtered dataframe once for reuse
    df_with_risk = df.copy()
    df_with_risk['risk_score'] = model.predict_proba(df_with_risk[FEATURE_COLS])[:, 1]
    
    # Create tabs for different EDA views
    eda_tab1, eda_tab2, eda_tab3 = st.tabs(["Distribution Overview", "Sensor Correlations", "Operational Analysis"])
    
    with eda_tab1:
        col1, col2 = st.columns(2)
        with col1:
            # Risk Distribution
            fig_risk_hist = px.histogram(
                df_with_risk, x="risk_score", nbins=40,
                title="Distribution of Risk Scores",
                labels={"risk_score": "Predicted Failure Risk", "count": "Number of Records"}
            )
            st.plotly_chart(fig_risk_hist, use_container_width=True)
        
        with col2:
            # Failure Label Distribution (if available)
            if "failure_24h" in df_with_risk.columns:
                failure_counts = df_with_risk['failure_24h'].value_counts().reset_index()
                failure_counts.columns = ['Failure Label', 'Count']
                fig_failure_pie = px.pie(
                    failure_counts, values='Count', names='Failure Label',
                    title="Proportion of Actual Failure Labels",
                    color='Failure Label',
                    color_discrete_map={0: 'green', 1: 'red'}
                )
                st.plotly_chart(fig_failure_pie, use_container_width=True)
            else:
                st.info("Ground truth failure labels not available in this dataset.")
                
        # Cycle Distribution by FD
        fig_cycle_box = px.box(
            df_with_risk, x="fd", y="time_cycles",
            title="Distribution of Time Cycles by Sub-dataset (FD)",
            labels={"time_cycles": "Time Cycles", "fd": "FD Sub-dataset"}
        )
        st.plotly_chart(fig_cycle_box, use_container_width=True)
    
    with eda_tab2:
        st.markdown("**Sensor Correlation Matrix**")
        sensor_cols_for_corr = [c for c in sensor_cols_all if c in df_with_risk.columns][:10]  # Limit to first 10 for clarity
        if len(sensor_cols_for_corr) > 1:
            sensor_corr = df_with_risk[sensor_cols_for_corr].corr()
            fig_sensor_heatmap = px.imshow(
                sensor_corr, 
                text_auto='.2f', 
                aspect="auto",
                title="Correlation Between Key Sensors",
                color_continuous_scale='RdBu_r'
            )
            st.plotly_chart(fig_sensor_heatmap, use_container_width=True)
        else:
            st.info("Need at least 2 sensors to compute correlation.")
            
        st.markdown("**Risk Correlation with Sensors**")
        # Check correlation of risk score with top N sensors
        top_sensors = sensor_cols_all[:5] 
        if top_sensors:
            risk_corr_data = []
            for sensor in top_sensors:
                if sensor in df_with_risk.columns:
                    corr_value = df_with_risk['risk_score'].corr(df_with_risk[sensor])
                    risk_corr_data.append({'Sensor': sensor, 'Correlation with Risk': corr_value})
            
            if risk_corr_data:
                risk_corr_df = pd.DataFrame(risk_corr_data).sort_values('Correlation with Risk', key=abs, ascending=False)
                fig_risk_corr = px.bar(
                    risk_corr_df, 
                    x='Correlation with Risk', 
                    y='Sensor', 
                    orientation='h',
                    title="Sensors Most Correlated with Model's Risk Score",
                    color='Correlation with Risk', 
                    color_continuous_scale='RdBu_r'
                )
                st.plotly_chart(fig_risk_corr, use_container_width=True)
    
    with eda_tab3:
        col1, col2 = st.columns(2)
        with col1:
            # Risk vs. Operational Setting
            if 'op_setting_1' in df_with_risk.columns:
                fig_op_risk = px.scatter(
                    df_with_risk, 
                    x='op_setting_1', 
                    y='risk_score',
                    title="Risk Score vs. Operational Setting 1",
                    trendline="lowess",
                    labels={"op_setting_1": "Op Setting 1", "risk_score": "Risk Score"}
                )
                st.plotly_chart(fig_op_risk, use_container_width=True)
        with col2:
            # Risk over Time (Averaged)
            avg_risk_over_cycles = df_with_risk.groupby('time_cycles')['risk_score'].mean().reset_index()
            fig_avg_risk = px.line(
                avg_risk_over_cycles, 
                x='time_cycles', 
                y='risk_score',
                title="Average Risk Score over Time (All Units)",
                labels={"time_cycles": "Time Cycle", "risk_score": "Avg. Risk Score"}
            )
            st.plotly_chart(fig_avg_risk, use_container_width=True)
            
    st.markdown("---")
    
    # --- Section 2: Fleet "Sentiment" / Health Analysis ---
    st.subheader("🧠 Fleet Health & Risk Sentiment Analysis")
    
    # Get the latest reading for every machine in the filtered dataset
    fleet_latest = df_with_risk.sort_values(['unit_id', 'time_cycles']).groupby('unit_id').tail(1).copy()
    
    # Define Health Status based on risk
    def get_health_status(risk_score, threshold=thr):
        if risk_score >= threshold:
            return "Critical"
        elif risk_score >= threshold * 0.7: # 70% of threshold
            return "Warning"
        else:
            return "Healthy"
    
    fleet_latest['Health Status'] = fleet_latest['risk_score'].apply(get_health_status)
    
    # Calculate Fleet Health Metrics
    total_machines = len(fleet_latest)
    critical_count = (fleet_latest['Health Status'] == 'Critical').sum()
    warning_count = (fleet_latest['Health Status'] == 'Warning').sum()
    healthy_count = (fleet_latest['Health Status'] == 'Healthy').sum()
    
    avg_fleet_risk = fleet_latest['risk_score'].mean()
    
    # Create a "Fleet Sentiment" score (0-100 scale, 100 is perfectly healthy)
    # Weight healthy machines more, penalize critical ones
    sentiment_score = (healthy_count * 1.0 + warning_count * 0.5 - critical_count * 1.0) / total_machines
    sentiment_score = max(0, min(100, (sentiment_score + 1) * 50))  # Scale from [-1,1] to [0,100]
    
    # Display Key Health Metrics
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Total Machines", total_machines)
    col2.metric("Avg. Fleet Risk", f"{avg_fleet_risk:.3f}")
    col3.metric("Fleet Health Score", f"{sentiment_score:.0f}/100")
    # Color the critical count metric
    critical_display = f"{critical_count} 🔴"
    col4.metric("Critical Machines", critical_display)
    
    # Health Status Pie Chart
    health_pie_data = fleet_latest['Health Status'].value_counts().reset_index()
    health_pie_data.columns = ['Status', 'Count']
    
    color_discrete_map = {"Healthy": "green", "Warning": "orange", "Critical": "red"}
    
    fig_health_pie = px.pie(
        health_pie_data, 
        values='Count', 
        names='Status',
        title="Fleet Health Distribution",
        color='Status',
        color_discrete_map=color_discrete_map
    )
    st.plotly_chart(fig_health_pie, use_container_width=True)
    
    # Top Risky Machines Table (Enhanced)
    st.subheader("🚨 Top Risky Machines (Requiring Attention)")
    topn = st.slider("Show top N risky machines", 5, min(50, len(fleet_latest)), 10, key="risky_slider")
    
    # Sort by risk descending and get top N
    top_risky = fleet_latest.sort_values("risk_score", ascending=False).head(topn)[
        ["unit_id", "fd", "time_cycles", "risk_score", "Health Status"]
    ].copy()
    # Format the risk score for better display
    top_risky['risk_score'] = top_risky['risk_score'].apply(lambda x: f'{x:.4f}')
    
    # Display the table with color formatting
    st.dataframe(
        top_risky.style.applymap(
            lambda x: 'color: red;' if x == 'Critical' else ('color: orange;' if x == 'Warning' else ''),
            subset=['Health Status']
        ),
        use_container_width=True,
        height=min(400, 35 * topn + 40) # Dynamic height based on rows
    )

    # Top risky machines
    st.subheader("Top Risky Machines")
    fleet = df.groupby("unit_id").tail(1).copy()  # last cycle per unit
    fleet["risk_score"] = model.predict_proba(fleet[FEATURE_COLS])[:,1]
    topn = st.slider("Show top N", 5, min(50, len(fleet)), 10)
    top_tbl = fleet.sort_values("risk_score", ascending=False).head(topn)[
        ["unit_id","fd","time_cycles","risk_score"]
    ]
    st.dataframe(top_tbl, use_container_width=True, height=320)

# ----- Explainability (simple model-based)
with tab_explain:
    st.subheader("Feature Importance / Coefficients")
    # LogisticRegression: coef_, DecisionTree: feature_importances_
    fig_imp = None
    if hasattr(model, "coef_"):
        coefs = model.coef_.ravel()
        imp_df = pd.DataFrame({"feature": FEATURE_COLS, "importance": np.abs(coefs), "signed": coefs})
        imp_df = imp_df.sort_values("importance", ascending=False).head(20)
        fig_imp = px.bar(imp_df, x="importance", y="feature", orientation="h",
                         title="Top 20 Absolute Coefficients")
    elif hasattr(model, "feature_importances_"):
        fi = model.feature_importances_
        imp_df = pd.DataFrame({"feature": FEATURE_COLS, "importance": fi})
        imp_df = imp_df.sort_values("importance", ascending=False).head(20)
        fig_imp = px.bar(imp_df, x="importance", y="feature", orientation="h",
                         title="Top 20 Feature Importances")
    if fig_imp:
        st.plotly_chart(fig_imp, use_container_width=True)
    else:
        st.info("Model does not expose feature importances/coefs.")

    # Simple per-record explanation: top contributing features by value*coef (for LR)
    if hasattr(model, "coef_"):
        st.subheader("Current Reading — Top Positive Contributors")
        row = unit_df_all.iloc[[-1]][FEATURE_COLS]
        contrib = row.values.ravel() * model.coef_.ravel()
        contrib_df = pd.DataFrame({"feature": FEATURE_COLS, "contribution": contrib})
        contrib_df = contrib_df.sort_values("contribution", ascending=False).head(10)
        fig_contrib = px.bar(contrib_df, x="contribution", y="feature", orientation="h",
                             title="Top +ve contributions to risk (last reading)")
        st.plotly_chart(fig_contrib, use_container_width=True)

# ----- Data / Export
with tab_data:
    st.subheader("Recent Records (Selected Unit)")
    st.dataframe(unit_df.tail(200), use_container_width=True, height=360)

    st.download_button(
        "⬇️ Download this table as CSV",
        unit_df.to_csv(index=False).encode("utf-8"),
        file_name=f"unit_{unit_id}_recent.csv",
        mime="text/csv",
        use_container_width=True,
    )

    if PDF_PATH.exists():
        st.markdown(f"📄 Report available: **{PDF_PATH.name}** (open from: `{PDF_PATH}`)")
    else:
        st.caption("No PDF report found yet.")



# ----------------------------
# Footer
# ----------------------------
st.markdown("---")
st.markdown(
    "<div class='footnote'>Built with Streamlit + Plotly • Optimized for the NASA C-MAPSS datasets • "
    "Try changing FD subset, sensors, time window and threshold in the sidebar.</div>",
    unsafe_allow_html=True,
)
