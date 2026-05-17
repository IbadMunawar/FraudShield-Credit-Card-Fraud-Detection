import os
import warnings
import gdown
warnings.filterwarnings("ignore")

from dotenv import load_dotenv
load_dotenv()

import numpy as np
import pandas as pd
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score,
    f1_score, roc_auc_score, roc_curve,
    confusion_matrix, precision_recall_curve
)
from xgboost import XGBClassifier
from imblearn.over_sampling import SMOTE
from imblearn.under_sampling import RandomUnderSampler
import shap

# ─── Page Config ────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="FraudShield — Credit Card Fraud Detection",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ─── Custom CSS ─────────────────────────────────────────────────────────────
st.markdown("""
<style>
  @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800&display=swap');

  html, body, [class*="css"] {
    font-family: 'Inter', sans-serif;
    background-color: #0F0F1A;
    color: #E0E0E0;
  }

  /* Sidebar */
  section[data-testid="stSidebar"] {
    background: linear-gradient(180deg, #0F0F1A 0%, #1A1A2E 100%);
    border-right: 1px solid #2D2D44;
  }
  section[data-testid="stSidebar"] .stRadio > label {
    font-size: 0.85rem;
    color: #A0A0C0;
    letter-spacing: 0.05em;
    text-transform: uppercase;
    font-weight: 600;
  }
  section[data-testid="stSidebar"] .stRadio div[role="radio"] {
    padding: 0.5rem 0.75rem;
    border-radius: 8px;
    margin-bottom: 4px;
    transition: background 0.2s;
  }
  section[data-testid="stSidebar"] .stRadio div[role="radio"]:hover {
    background: rgba(233,69,96,0.15);
  }

  /* KPI Cards */
  .kpi-card {
    background: linear-gradient(135deg, #1A1A2E 0%, #16213E 100%);
    border: 1px solid #2D2D44;
    border-radius: 16px;
    padding: 1.5rem;
    text-align: center;
    transition: transform 0.2s, box-shadow 0.2s;
  }
  .kpi-card:hover {
    transform: translateY(-4px);
    box-shadow: 0 8px 32px rgba(233,69,96,0.2);
  }
  .kpi-value {
    font-size: 2.2rem;
    font-weight: 800;
    line-height: 1.1;
    margin: 0.25rem 0;
  }
  .kpi-label {
    font-size: 0.82rem;
    color: #8888AA;
    text-transform: uppercase;
    letter-spacing: 0.08em;
    font-weight: 500;
  }
  .kpi-icon { font-size: 1.6rem; margin-bottom: 0.25rem; }
  .red   { color: #E94560; }
  .green { color: #00B894; }
  .blue  { color: #74B9FF; }
  .gold  { color: #FDCB6E; }

  /* Section headers */
  .section-header {
    font-size: 1.5rem;
    font-weight: 700;
    color: #FFFFFF;
    border-left: 4px solid #E94560;
    padding-left: 0.75rem;
    margin: 1.5rem 0 1rem 0;
  }

  /* Best model badge */
  .best-badge {
    background: linear-gradient(90deg, #E94560, #c0392b);
    color: white;
    padding: 0.2rem 0.7rem;
    border-radius: 20px;
    font-size: 0.75rem;
    font-weight: 600;
    letter-spacing: 0.05em;
  }

  /* Metric table */
  .metric-table {
    background: #1A1A2E;
    border: 1px solid #2D2D44;
    border-radius: 12px;
    padding: 1rem;
  }

  /* Prediction box */
  .pred-box-fraud {
    background: linear-gradient(135deg, rgba(233,69,96,0.2), rgba(233,69,96,0.05));
    border: 2px solid #E94560;
    border-radius: 16px;
    padding: 2rem;
    text-align: center;
  }
  .pred-box-legit {
    background: linear-gradient(135deg, rgba(0,184,148,0.2), rgba(0,184,148,0.05));
    border: 2px solid #00B894;
    border-radius: 16px;
    padding: 2rem;
    text-align: center;
  }
  .pred-title { font-size: 1.8rem; font-weight: 800; margin-bottom: 0.5rem; }
  .pred-prob  { font-size: 1.1rem; color: #A0A0C0; }

  /* Dividers */
  hr { border-color: #2D2D44; margin: 1.5rem 0; }

  /* Hide Streamlit branding */
  #MainMenu, footer { visibility: hidden; }
  .block-container { padding-top: 1.5rem; }
</style>
""", unsafe_allow_html=True)

# ─── Constants ───────────────────────────────────────────────────────────────
DATASET_PATH = os.getenv("DATASET_PATH", "creditcard.csv")
PLOTLY_TEMPLATE = "plotly_dark"
FRAUD_COLOR  = "#E94560"
LEGIT_COLOR  = "#00B894"
BLUE_COLOR   = "#74B9FF"
BG_COLOR     = "#0F0F1A"
CARD_BG      = "#1A1A2E"

FEATURE_COLS = ["Amount"] + [f"V{i}" for i in range(1, 29)]
TARGET_COL   = "Class"

# ─── Data Loading ────────────────────────────────────────────────────────────
@st.cache_data(show_spinner="Loading dataset…")
def load_data():
    if not os.path.exists(DATASET_PATH):
        url = "https://drive.google.com/uc?id=1u_k3uKbLEmCFb0u4-M0nXpF7p2unY0zx"
        with st.spinner("📥 creditcard.csv not found locally — downloading from Google Drive…"):
            gdown.download(url, DATASET_PATH, quiet=False)
    df = pd.read_csv(DATASET_PATH)
    df["Hour"] = (df["Time"] // 3600) % 24
    df["Amount_log"] = np.log1p(df["Amount"])
    return df

# ─── Model Training ─────────────────────────────────────────────────────────
@st.cache_resource(show_spinner="Training models (first run only)…")
def train_models(df_hash: str):
    df = load_data()

    # --- Prepare features ---
    X = df[FEATURE_COLS]
    y = df[TARGET_COL]

    # --- Scale Amount ---
    scaler = StandardScaler()
    X = X.copy()
    X["Amount"] = scaler.fit_transform(X[["Amount"]])

    # --- Train/Test split ---
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    # --- Undersample majority class to 50k before SMOTE ---
    n_fraud = y_train.sum()
    n_legit_target = min(50_000, len(y_train[y_train == 0]))
    rus = RandomUnderSampler(
        sampling_strategy={0: n_legit_target, 1: int(n_fraud)},
        random_state=42
    )
    X_res, y_res = rus.fit_resample(X_train, y_train)

    # --- SMOTE to balance classes ---
    smote = SMOTE(random_state=42, k_neighbors=5)
    X_bal, y_bal = smote.fit_resample(X_res, y_res)

    # --- Define models ---
    models_def = {
        "Logistic Regression": LogisticRegression(max_iter=1000, random_state=42, class_weight="balanced"),
        "Random Forest":       RandomForestClassifier(n_estimators=150, random_state=42, n_jobs=-1),
        "XGBoost":             XGBClassifier(n_estimators=150, random_state=42,
                                             use_label_encoder=False, eval_metric="logloss",
                                             tree_method="hist"),
    }

    results = {}
    for name, model in models_def.items():
        model.fit(X_bal, y_bal)
        y_pred  = model.predict(X_test)
        y_prob  = model.predict_proba(X_test)[:, 1]
        fpr, tpr, thresholds_roc = roc_curve(y_test, y_prob)
        prec_arr, rec_arr, thresholds_pr = precision_recall_curve(y_test, y_prob)
        results[name] = {
            "model":      model,
            "y_test":     y_test,
            "y_pred":     y_pred,
            "y_prob":     y_prob,
            "accuracy":   accuracy_score(y_test, y_pred),
            "precision":  precision_score(y_test, y_pred, zero_division=0),
            "recall":     recall_score(y_test, y_pred, zero_division=0),
            "f1":         f1_score(y_test, y_pred, zero_division=0),
            "auc":        roc_auc_score(y_test, y_prob),
            "fpr":        fpr,
            "tpr":        tpr,
            "prec_arr":   prec_arr,
            "rec_arr":    rec_arr,
            "thresholds_pr": thresholds_pr,
            "cm":         confusion_matrix(y_test, y_pred),
        }

    # Best model by AUC-ROC
    best_name = max(results, key=lambda k: results[k]["auc"])

    # SHAP explainer for best model
    best_model = results[best_name]["model"]
    if best_name == "Logistic Regression":
        explainer = shap.LinearExplainer(best_model, X_bal)
    else:
        explainer = shap.TreeExplainer(best_model)

    return results, best_name, explainer, scaler, X_test, y_test, list(X.columns)

# ─── Helpers ─────────────────────────────────────────────────────────────────
def kpi_card(icon, label, value, color_class="blue"):
    return f"""
    <div class="kpi-card">
      <div class="kpi-icon">{icon}</div>
      <div class="kpi-value {color_class}">{value}</div>
      <div class="kpi-label">{label}</div>
    </div>"""

def plotly_defaults(fig, height=400):
    fig.update_layout(
        template=PLOTLY_TEMPLATE,
        paper_bgcolor=CARD_BG,
        plot_bgcolor=CARD_BG,
        font=dict(family="Inter", color="#C0C0D8"),
        height=height,
        margin=dict(l=20, r=20, t=40, b=20),
    )
    return fig

# ═══════════════════════════════════════════════════════════════════════════════
# PAGE 1 — OVERVIEW
# ═══════════════════════════════════════════════════════════════════════════════
def page_overview(df):
    st.markdown(
        "<span style='font-size:2rem;font-weight:800;color:#FFFFFF;'>📊 Dashboard Overview</span>",
        unsafe_allow_html=True,
    )
    st.markdown(
        "<p style='color:#8888AA;margin-top:0;'>Real-time summary of the credit card transaction dataset</p>",
        unsafe_allow_html=True,
    )
    st.markdown("---")

    total          = len(df)
    fraud_count    = int(df[TARGET_COL].sum())
    legit_count    = total - fraud_count
    fraud_rate     = fraud_count / total * 100
    avg_fraud_amt  = df[df[TARGET_COL] == 1]["Amount"].mean()
    avg_legit_amt  = df[df[TARGET_COL] == 0]["Amount"].mean()
    max_fraud      = df[df[TARGET_COL] == 1]["Amount"].max()
    total_loss     = df[df[TARGET_COL] == 1]["Amount"].sum()

    # ── KPI Row 1 ──────────────────────────────────────────────────────────────
    c1, c2, c3, c4 = st.columns(4)
    with c1:
        st.markdown(kpi_card("💳", "Total Transactions", f"{total:,}", "blue"), unsafe_allow_html=True)
    with c2:
        st.markdown(kpi_card("🚨", "Fraud Cases", f"{fraud_count:,}", "red"), unsafe_allow_html=True)
    with c3:
        st.markdown(kpi_card("📈", "Fraud Rate", f"{fraud_rate:.3f}%", "gold"), unsafe_allow_html=True)
    with c4:
        st.markdown(kpi_card("💰", "Avg Fraud Amount", f"${avg_fraud_amt:.2f}", "red"), unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    # ── KPI Row 2 ──────────────────────────────────────────────────────────────
    c5, c6, c7, c8 = st.columns(4)
    with c5:
        st.markdown(kpi_card("✅", "Legitimate Cases", f"{legit_count:,}", "green"), unsafe_allow_html=True)
    with c6:
        st.markdown(kpi_card("💵", "Avg Legit Amount", f"${avg_legit_amt:.2f}", "green"), unsafe_allow_html=True)
    with c7:
        st.markdown(kpi_card("⚠️", "Max Fraud Amount", f"${max_fraud:.2f}", "gold"), unsafe_allow_html=True)
    with c8:
        st.markdown(kpi_card("📉", "Total Fraud Value", f"${total_loss:,.0f}", "red"), unsafe_allow_html=True)

    st.markdown("---")
    st.markdown("<div class='section-header'>Class Distribution</div>", unsafe_allow_html=True)

    left, right = st.columns(2)

    with left:
        fig_pie = go.Figure(data=[go.Pie(
            labels=["Legitimate", "Fraud"],
            values=[legit_count, fraud_count],
            hole=0.62,
            marker_colors=[LEGIT_COLOR, FRAUD_COLOR],
            textinfo="label+percent",
        )])
        fig_pie.add_annotation(
            text=f"<b>{fraud_rate:.2f}%</b><br>Fraud Rate",
            x=0.5, y=0.5, showarrow=False,
            font=dict(size=16, color=FRAUD_COLOR),
        )
        fig_pie.update_layout(
            title="Fraud vs Legitimate Transactions",
            showlegend=True,
            legend=dict(orientation="h", y=-0.05),
        )
        st.plotly_chart(plotly_defaults(fig_pie, 380), use_container_width=True)

    with right:
        fig_bar = go.Figure()
        fig_bar.add_trace(go.Bar(
            x=["Legitimate (0)", "Fraud (1)"],
            y=[legit_count, fraud_count],
            marker_color=[LEGIT_COLOR, FRAUD_COLOR],
            text=[f"{legit_count:,}", f"{fraud_count:,}"],
            textposition="outside",
        ))
        fig_bar.update_layout(
            title="Class Imbalance (Log Scale)",
            yaxis_type="log",
            yaxis_title="Count (log scale)",
            showlegend=False,
        )
        st.plotly_chart(plotly_defaults(fig_bar, 380), use_container_width=True)

    st.markdown("---")
    st.markdown("<div class='section-header'>Transaction Amount Distribution</div>", unsafe_allow_html=True)

    fig_amount = go.Figure()
    for cls, color, label in [(0, LEGIT_COLOR, "Legitimate"), (1, FRAUD_COLOR, "Fraud")]:
        subset = df[df[TARGET_COL] == cls]["Amount"]
        fig_amount.add_trace(go.Histogram(
            x=subset, name=label,
            marker_color=color, opacity=0.72, nbinsx=80,
        ))
    fig_amount.update_layout(
        title="Transaction Amount Distribution by Class",
        xaxis_title="Amount ($)", yaxis_title="Count",
        barmode="overlay",
        legend=dict(orientation="h", y=1.1),
    )
    st.plotly_chart(plotly_defaults(fig_amount, 360), use_container_width=True)


# ═══════════════════════════════════════════════════════════════════════════════
# PAGE 2 — EDA & ANALYSIS
# ═══════════════════════════════════════════════════════════════════════════════
def page_eda(df):
    st.markdown(
        "<span style='font-size:2rem;font-weight:800;color:#FFFFFF;'>🔍 EDA & Analysis</span>",
        unsafe_allow_html=True,
    )
    st.markdown(
        "<p style='color:#8888AA;'>Deep-dive exploratory data analysis of transaction patterns</p>",
        unsafe_allow_html=True,
    )
    st.markdown("---")

    # ── Amount by class ────────────────────────────────────────────────────────
    st.markdown("<div class='section-header'>Amount Distribution by Class</div>", unsafe_allow_html=True)
    col1, col2 = st.columns(2)

    with col1:
        fig_box = go.Figure()
        for cls, color, label in [(0, LEGIT_COLOR, "Legitimate"), (1, FRAUD_COLOR, "Fraud")]:
            fig_box.add_trace(go.Box(
                y=df[df[TARGET_COL] == cls]["Amount"],
                name=label, marker_color=color, boxmean="sd",
            ))
        fig_box.update_layout(title="Amount Box Plot (log scale)", yaxis_type="log", yaxis_title="Amount ($)")
        st.plotly_chart(plotly_defaults(fig_box, 380), use_container_width=True)

    with col2:
        fig_violin = go.Figure()
        for cls, color, label in [(0, LEGIT_COLOR, "Legitimate"), (1, FRAUD_COLOR, "Fraud")]:
            fig_violin.add_trace(go.Violin(
                y=df[df[TARGET_COL] == cls]["Amount_log"],
                name=label, box_visible=True,
                meanline_visible=True, line_color=color, fillcolor=color, opacity=0.5,
            ))
        fig_violin.update_layout(title="Log(Amount+1) Violin Plot", yaxis_title="log(Amount+1)")
        st.plotly_chart(plotly_defaults(fig_violin, 380), use_container_width=True)

    st.markdown("---")

    # ── Time-based patterns ────────────────────────────────────────────────────
    st.markdown("<div class='section-header'>Time-Based Fraud Patterns</div>", unsafe_allow_html=True)
    col3, col4 = st.columns(2)

    hourly = df.groupby(["Hour", TARGET_COL]).size().reset_index(name="count")

    with col3:
        fraud_h = hourly[hourly[TARGET_COL] == 1]
        legit_h = hourly[hourly[TARGET_COL] == 0]
        fig_hour = go.Figure()
        fig_hour.add_trace(go.Scatter(
            x=legit_h["Hour"], y=legit_h["count"],
            name="Legitimate", line=dict(color=LEGIT_COLOR, width=2.5),
            fill="tozeroy", fillcolor="rgba(0,184,148,0.1)",
        ))
        fig_hour.add_trace(go.Scatter(
            x=fraud_h["Hour"], y=fraud_h["count"],
            name="Fraud", line=dict(color=FRAUD_COLOR, width=2.5),
            fill="tozeroy", fillcolor="rgba(233,69,96,0.15)",
            yaxis="y2",
        ))
        fig_hour.update_layout(
            title="Transactions by Hour of Day",
            xaxis_title="Hour",
            yaxis_title="Legitimate Count",
            yaxis2=dict(title="Fraud Count", overlaying="y", side="right", showgrid=False),
            legend=dict(orientation="h", y=1.1),
        )
        st.plotly_chart(plotly_defaults(fig_hour, 380), use_container_width=True)

    with col4:
        pivot = df.groupby("Hour")[TARGET_COL].agg(["sum", "count"]).reset_index()
        pivot["fraud_rate"] = pivot["sum"] / pivot["count"] * 100
        fig_rate = go.Figure(go.Bar(
            x=pivot["Hour"], y=pivot["fraud_rate"],
            marker=dict(
                color=pivot["fraud_rate"],
                colorscale=[[0, LEGIT_COLOR], [1, FRAUD_COLOR]],
                showscale=True,
                colorbar=dict(title="Fraud %"),
            ),
        ))
        fig_rate.update_layout(title="Fraud Rate by Hour (%)", xaxis_title="Hour", yaxis_title="Fraud Rate (%)")
        st.plotly_chart(plotly_defaults(fig_rate, 380), use_container_width=True)

    st.markdown("---")

    # ── Correlation Heatmap ────────────────────────────────────────────────────
    st.markdown("<div class='section-header'>Correlation Heatmap (Top 15 Features)</div>", unsafe_allow_html=True)
    sample_df = df.sample(min(5000, len(df)), random_state=42)
    top_corr_cols = (
        sample_df[FEATURE_COLS + [TARGET_COL]]
        .corr()[TARGET_COL]
        .abs()
        .sort_values(ascending=False)
        .head(15)
        .index.tolist()
    )
    corr_matrix = sample_df[top_corr_cols].corr()
    fig_heat = go.Figure(go.Heatmap(
        z=corr_matrix.values,
        x=corr_matrix.columns.tolist(),
        y=corr_matrix.index.tolist(),
        colorscale=[[0, "#00B894"], [0.5, CARD_BG], [1, "#E94560"]],
        zmid=0,
        text=[[f"{v:.2f}" for v in row] for row in corr_matrix.values],
        texttemplate="%{text}",
        textfont=dict(size=9),
    ))
    fig_heat.update_layout(title="Feature Correlation Matrix (sampled 5 000 rows)")
    st.plotly_chart(plotly_defaults(fig_heat, 520), use_container_width=True)

    st.markdown("---")

    # ── Top PCA features ──────────────────────────────────────────────────────
    st.markdown("<div class='section-header'>Top PCA Features by Fraud Correlation</div>", unsafe_allow_html=True)
    v_cols = [f"V{i}" for i in range(1, 29)]
    corr_class = df[v_cols + [TARGET_COL]].corr()[TARGET_COL].drop(TARGET_COL).abs().sort_values(ascending=False)
    top6 = corr_class.head(6).index.tolist()

    fig_pca = make_subplots(rows=2, cols=3, subplot_titles=top6)
    for idx, col in enumerate(top6):
        r, c = divmod(idx, 3)
        for cls, color, label in [(0, LEGIT_COLOR, "Legit"), (1, FRAUD_COLOR, "Fraud")]:
            fig_pca.add_trace(go.Histogram(
                x=df[df[TARGET_COL] == cls][col],
                name=label if idx == 0 else None,
                marker_color=color, opacity=0.65, nbinsx=60,
                showlegend=(idx == 0),
            ), row=r + 1, col=c + 1)
    fig_pca.update_layout(
        title_text="Top 6 PCA Features: Fraud vs Legitimate",
        barmode="overlay",
        legend=dict(orientation="h", y=-0.08),
    )
    st.plotly_chart(plotly_defaults(fig_pca, 600), use_container_width=True)


# ═══════════════════════════════════════════════════════════════════════════════
# PAGE 3 — ML MODELS
# ═══════════════════════════════════════════════════════════════════════════════
def page_ml_models(results, best_name, feature_cols):
    st.markdown(
        "<span style='font-size:2rem;font-weight:800;color:#FFFFFF;'>🤖 ML Models</span>",
        unsafe_allow_html=True,
    )
    st.markdown(
        "<p style='color:#8888AA;'>Model training with SMOTE balancing — compare performance across 3 algorithms</p>",
        unsafe_allow_html=True,
    )
    st.markdown("---")

    model_names = list(results.keys())

    # ── Metrics Table ─────────────────────────────────────────────────────────
    st.markdown("<div class='section-header'>Model Performance Metrics</div>", unsafe_allow_html=True)

    metrics_data = []
    for name in model_names:
        r = results[name]
        badge = " 🏆 Best" if name == best_name else ""
        metrics_data.append({
            "Model": name + badge,
            "Accuracy":  f"{r['accuracy']:.4f}",
            "Precision": f"{r['precision']:.4f}",
            "Recall":    f"{r['recall']:.4f}",
            "F1 Score":  f"{r['f1']:.4f}",
            "AUC-ROC":   f"{r['auc']:.4f}",
        })
    metrics_df = pd.DataFrame(metrics_data)
    st.dataframe(
        metrics_df.style
            .set_properties(**{"background-color": "#1A1A2E", "color": "#E0E0E0", "border-color": "#2D2D44"})
            .highlight_max(axis=0, subset=["Accuracy", "Precision", "Recall", "F1 Score", "AUC-ROC"],
                           props="background-color: rgba(233,69,96,0.25); font-weight: bold;"),
        use_container_width=True,
        hide_index=True,
    )

    st.markdown("---")

    # ── ROC Curves ────────────────────────────────────────────────────────────
    st.markdown("<div class='section-header'>ROC Curves</div>", unsafe_allow_html=True)
    COLORS = [FRAUD_COLOR, LEGIT_COLOR, BLUE_COLOR]
    fig_roc = go.Figure()
    fig_roc.add_shape(type="line", x0=0, y0=0, x1=1, y1=1,
                      line=dict(dash="dot", color="#555577", width=1))
    for idx, name in enumerate(model_names):
        r = results[name]
        fig_roc.add_trace(go.Scatter(
            x=r["fpr"], y=r["tpr"],
            name=f"{name} (AUC={r['auc']:.3f})",
            line=dict(color=COLORS[idx % len(COLORS)], width=2.5),
            mode="lines",
        ))
    fig_roc.update_layout(
        title="ROC Curves — All Models",
        xaxis_title="False Positive Rate",
        yaxis_title="True Positive Rate",
        legend=dict(orientation="h", y=-0.15),
    )
    st.plotly_chart(plotly_defaults(fig_roc, 450), use_container_width=True)

    st.markdown("---")

    # ── Confusion Matrices ────────────────────────────────────────────────────
    st.markdown("<div class='section-header'>Confusion Matrices</div>", unsafe_allow_html=True)
    cm_cols = st.columns(3)
    for idx, name in enumerate(model_names):
        cm = results[name]["cm"]
        labels = ["Legit (0)", "Fraud (1)"]
        fig_cm = go.Figure(go.Heatmap(
            z=cm,
            x=labels, y=labels,
            colorscale=[[0, CARD_BG], [1, FRAUD_COLOR]],
            text=[[str(v) for v in row] for row in cm],
            texttemplate="<b>%{text}</b>",
            textfont=dict(size=16),
            showscale=False,
        ))
        badge = " 🏆" if name == best_name else ""
        fig_cm.update_layout(
            title=f"{name}{badge}",
            xaxis_title="Predicted", yaxis_title="Actual",
        )
        with cm_cols[idx]:
            st.plotly_chart(plotly_defaults(fig_cm, 300), use_container_width=True)

    st.markdown("---")

    # ── Feature Importance ───────────────────────────────────────────────────
    st.markdown(f"<div class='section-header'>Feature Importance — {best_name} (Best Model)</div>", unsafe_allow_html=True)
    best_model = results[best_name]["model"]

    if hasattr(best_model, "feature_importances_"):
        importances = best_model.feature_importances_
    elif hasattr(best_model, "coef_"):
        importances = np.abs(best_model.coef_[0])
    else:
        importances = np.ones(len(feature_cols))

    feat_df = pd.DataFrame({"Feature": feature_cols, "Importance": importances})
    feat_df = feat_df.sort_values("Importance", ascending=True).tail(20)

    fig_fi = go.Figure(go.Bar(
        x=feat_df["Importance"],
        y=feat_df["Feature"],
        orientation="h",
        marker=dict(
            color=feat_df["Importance"],
            colorscale=[[0, "#2D2D44"], [1, FRAUD_COLOR]],
            showscale=False,
        ),
    ))
    fig_fi.update_layout(
        title=f"Top 20 Feature Importances ({best_name})",
        xaxis_title="Importance Score",
        yaxis_title="Feature",
    )
    st.plotly_chart(plotly_defaults(fig_fi, 500), use_container_width=True)

    # Best model callout
    st.markdown(f"""
    <div style='background:linear-gradient(135deg,rgba(233,69,96,0.15),rgba(233,69,96,0.05));
         border:1px solid #E94560;border-radius:12px;padding:1.2rem 1.5rem;margin-top:1rem;'>
      <span style='font-size:1.1rem;font-weight:700;color:#FFFFFF;'>🏆 Best Model: {best_name}</span><br>
      <span style='color:#A0A0C0;font-size:0.9rem;'>
        AUC-ROC: <b style='color:#E94560'>{results[best_name]['auc']:.4f}</b> &nbsp;|&nbsp;
        F1: <b style='color:#E94560'>{results[best_name]['f1']:.4f}</b> &nbsp;|&nbsp;
        Recall: <b style='color:#E94560'>{results[best_name]['recall']:.4f}</b>
      </span>
    </div>
    """, unsafe_allow_html=True)


# ═══════════════════════════════════════════════════════════════════════════════
# PAGE 4 — FRAUD PREDICTOR
# ═══════════════════════════════════════════════════════════════════════════════
def page_predictor(results, best_name, explainer, scaler, feature_cols):
    st.markdown(
        "<span style='font-size:2rem;font-weight:800;color:#FFFFFF;'>🎯 Fraud Predictor</span>",
        unsafe_allow_html=True,
    )
    st.markdown(
        "<p style='color:#8888AA;'>Adjust feature values below and get an instant fraud probability with SHAP explanation</p>",
        unsafe_allow_html=True,
    )
    st.markdown("---")

    # ── Input Panel ───────────────────────────────────────────────────────────
    st.markdown("<div class='section-header'>Input Features</div>", unsafe_allow_html=True)

    input_vals = {}

    with st.expander("💵 Amount", expanded=True):
        input_vals["Amount"] = st.slider("Transaction Amount ($)", 0.0, 25000.0, 100.0, step=0.5)

    v_cols = [f"V{i}" for i in range(1, 29)]
    cols_per_row = 4
    rows = [v_cols[i:i+cols_per_row] for i in range(0, len(v_cols), cols_per_row)]

    with st.expander("📊 PCA Features (V1–V28)", expanded=False):
        for row in rows:
            slider_cols = st.columns(len(row))
            for sc, vcol in zip(slider_cols, row):
                with sc:
                    input_vals[vcol] = st.slider(vcol, -30.0, 30.0, 0.0, step=0.01, key=f"slider_{vcol}")

    # ── Prediction ────────────────────────────────────────────────────────────
    st.markdown("---")
    st.markdown("<div class='section-header'>Prediction Result</div>", unsafe_allow_html=True)

    # Build input dataframe
    input_row = pd.DataFrame([{col: input_vals.get(col, 0.0) for col in feature_cols}])

    # Scale Amount
    input_scaled = input_row.copy()
    input_scaled["Amount"] = scaler.transform(input_row[["Amount"]])

    best_model = results[best_name]["model"]
    prob = best_model.predict_proba(input_scaled)[0][1]
    pred = int(prob >= 0.5)

    # Display result
    res_col, gauge_col = st.columns([1, 1])

    with res_col:
        if pred == 1:
            st.markdown(f"""
            <div class='pred-box-fraud'>
              <div class='pred-title' style='color:#E94560;'>🚨 FRAUD DETECTED</div>
              <div class='pred-prob'>Fraud Probability: <b style='color:#E94560;font-size:1.6rem'>{prob*100:.1f}%</b></div>
              <br><span style='color:#A0A0C0;font-size:0.85rem;'>Model: {best_name}</span>
            </div>
            """, unsafe_allow_html=True)
        else:
            st.markdown(f"""
            <div class='pred-box-legit'>
              <div class='pred-title' style='color:#00B894;'>✅ LEGITIMATE</div>
              <div class='pred-prob'>Fraud Probability: <b style='color:#00B894;font-size:1.6rem'>{prob*100:.1f}%</b></div>
              <br><span style='color:#A0A0C0;font-size:0.85rem;'>Model: {best_name}</span>
            </div>
            """, unsafe_allow_html=True)

    with gauge_col:
        fig_gauge = go.Figure(go.Indicator(
            mode="gauge+number+delta",
            value=prob * 100,
            title={"text": "Fraud Probability (%)", "font": {"size": 16, "color": "#C0C0D8"}},
            gauge={
                "axis": {"range": [0, 100], "tickcolor": "#555577"},
                "bar": {"color": FRAUD_COLOR if pred == 1 else LEGIT_COLOR},
                "bgcolor": CARD_BG,
                "borderwidth": 1,
                "bordercolor": "#2D2D44",
                "steps": [
                    {"range": [0, 30], "color": "rgba(0,184,148,0.2)"},
                    {"range": [30, 70], "color": "rgba(253,203,110,0.15)"},
                    {"range": [70, 100], "color": "rgba(233,69,96,0.2)"},
                ],
                "threshold": {"line": {"color": "#FFFFFF", "width": 3}, "value": 50},
            },
            number={"suffix": "%", "font": {"size": 36, "color": FRAUD_COLOR if pred == 1 else LEGIT_COLOR}},
        ))
        fig_gauge.update_layout(paper_bgcolor=CARD_BG, font=dict(family="Inter"), height=300, margin=dict(t=20, b=10))
        st.plotly_chart(fig_gauge, use_container_width=True)

    # ── SHAP Explanation ──────────────────────────────────────────────────────
    st.markdown("---")
    st.markdown("<div class='section-header'>SHAP Explanation — Why This Prediction?</div>", unsafe_allow_html=True)

    with st.spinner("Computing SHAP values…"):
        shap_vals = explainer.shap_values(input_scaled)
        # For tree models shap_vals may be a list [class0, class1]
        if isinstance(shap_vals, list):
            sv = shap_vals[1][0]
        else:
            sv = shap_vals[0]

        shap_df = pd.DataFrame({
            "Feature": feature_cols,
            "SHAP Value": sv,
        }).sort_values("SHAP Value", key=abs, ascending=True).tail(15)

        colors = [FRAUD_COLOR if v > 0 else LEGIT_COLOR for v in shap_df["SHAP Value"]]
        fig_shap = go.Figure(go.Bar(
            x=shap_df["SHAP Value"],
            y=shap_df["Feature"],
            orientation="h",
            marker_color=colors,
            text=[f"{v:+.4f}" for v in shap_df["SHAP Value"]],
            textposition="outside",
        ))
        fig_shap.update_layout(
            title="SHAP Feature Contributions (Top 15) — Red=Pushes Toward Fraud, Green=Away",
            xaxis_title="SHAP Value",
            yaxis_title="Feature",
        )
        st.plotly_chart(plotly_defaults(fig_shap, 500), use_container_width=True)

    st.markdown("""
    <div style='background:#1A1A2E;border:1px solid #2D2D44;border-radius:10px;padding:1rem;margin-top:0.5rem;'>
      <b style='color:#FDCB6E;'>ℹ️ How to read SHAP:</b>
      <span style='color:#A0A0C0;font-size:0.88rem;'>
        Each bar shows how much a feature pushed the model's output toward fraud (red/positive)
        or away from fraud (green/negative). Longer bars = stronger influence.
      </span>
    </div>
    """, unsafe_allow_html=True)


# ═══════════════════════════════════════════════════════════════════════════════
# PAGE 5 — BUSINESS IMPACT
# ═══════════════════════════════════════════════════════════════════════════════
def page_business_impact(df, results, best_name):
    st.markdown(
        "<span style='font-size:2rem;font-weight:800;color:#FFFFFF;'>💼 Business Impact</span>",
        unsafe_allow_html=True,
    )
    st.markdown(
        "<p style='color:#8888AA;'>Quantify financial savings and tune the decision threshold to match business needs</p>",
        unsafe_allow_html=True,
    )
    st.markdown("---")

    r = results[best_name]
    y_test = r["y_test"]
    y_prob = r["y_prob"]

    avg_fraud_amt = df[df[TARGET_COL] == 1]["Amount"].mean()
    avg_legit_amt = df[df[TARGET_COL] == 0]["Amount"].mean()

    # ── Threshold Tuner ───────────────────────────────────────────────────────
    st.markdown("<div class='section-header'>Decision Threshold Tuning</div>", unsafe_allow_html=True)
    threshold = st.slider(
        "Classification Threshold",
        min_value=0.01, max_value=0.99, value=0.5, step=0.01,
        help="Lower threshold → catch more fraud (higher recall, lower precision). "
             "Higher threshold → fewer false alarms (higher precision, lower recall).",
    )

    y_pred_thresh = (y_prob >= threshold).astype(int)
    tp = int(((y_pred_thresh == 1) & (y_test == 1)).sum())
    fp = int(((y_pred_thresh == 1) & (y_test == 0)).sum())
    tn = int(((y_pred_thresh == 0) & (y_test == 0)).sum())
    fn = int(((y_pred_thresh == 0) & (y_test == 1)).sum())

    prec  = tp / (tp + fp) if (tp + fp) > 0 else 0
    rec   = tp / (tp + fn) if (tp + fn) > 0 else 0
    f1_t  = 2 * prec * rec / (prec + rec) if (prec + rec) > 0 else 0
    fpr_t = fp / (fp + tn) if (fp + tn) > 0 else 0

    # Estimate fraud amount in test set
    fraud_saved     = tp * avg_fraud_amt
    false_alarm_cost = fp * avg_legit_amt * 0.02   # 2% friction cost per false alarm
    net_savings     = fraud_saved - false_alarm_cost

    # KPI cards
    k1, k2, k3, k4 = st.columns(4)
    with k1:
        st.markdown(kpi_card("✅", "True Positives (Caught Fraud)", f"{tp:,}", "red"), unsafe_allow_html=True)
    with k2:
        st.markdown(kpi_card("❌", "False Positives (False Alarms)", f"{fp:,}", "gold"), unsafe_allow_html=True)
    with k3:
        st.markdown(kpi_card("💰", "Est. Fraud Savings", f"${fraud_saved:,.0f}", "green"), unsafe_allow_html=True)
    with k4:
        st.markdown(kpi_card("📊", "Net Financial Benefit", f"${net_savings:,.0f}", "green" if net_savings > 0 else "red"), unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    k5, k6, k7, k8 = st.columns(4)
    with k5:
        st.markdown(kpi_card("🎯", "Precision @ Threshold", f"{prec:.3f}", "blue"), unsafe_allow_html=True)
    with k6:
        st.markdown(kpi_card("🔍", "Recall @ Threshold", f"{rec:.3f}", "blue"), unsafe_allow_html=True)
    with k7:
        st.markdown(kpi_card("⚡", "F1 @ Threshold", f"{f1_t:.3f}", "blue"), unsafe_allow_html=True)
    with k8:
        st.markdown(kpi_card("📉", "False Positive Rate", f"{fpr_t:.3f}", "gold"), unsafe_allow_html=True)

    st.markdown("---")

    # ── Precision-Recall Curve ─────────────────────────────────────────────────
    st.markdown("<div class='section-header'>Precision-Recall Tradeoff</div>", unsafe_allow_html=True)

    prec_arr = r["prec_arr"]
    rec_arr  = r["rec_arr"]
    thr_arr  = r["thresholds_pr"]

    # find closest threshold index
    if len(thr_arr) > 0:
        closest_idx = int(np.argmin(np.abs(thr_arr - threshold)))
        marker_prec = float(prec_arr[closest_idx])
        marker_rec  = float(rec_arr[closest_idx])
    else:
        marker_prec = marker_rec = 0.0

    fig_pr = go.Figure()
    fig_pr.add_trace(go.Scatter(
        x=rec_arr, y=prec_arr,
        mode="lines",
        name=f"{best_name} PR Curve",
        line=dict(color=FRAUD_COLOR, width=2.5),
        fill="tozeroy",
        fillcolor="rgba(233,69,96,0.1)",
    ))
    fig_pr.add_trace(go.Scatter(
        x=[marker_rec], y=[marker_prec],
        mode="markers",
        name=f"Threshold = {threshold:.2f}",
        marker=dict(color="#FDCB6E", size=14, symbol="diamond",
                    line=dict(color="#FFFFFF", width=2)),
    ))
    fig_pr.update_layout(
        title=f"Precision-Recall Curve — {best_name}",
        xaxis_title="Recall",
        yaxis_title="Precision",
        legend=dict(orientation="h", y=1.1),
    )
    st.plotly_chart(plotly_defaults(fig_pr, 420), use_container_width=True)

    st.markdown("---")

    # ── Threshold Sweep: Savings vs Threshold ─────────────────────────────────
    st.markdown("<div class='section-header'>Financial Savings Across Thresholds</div>", unsafe_allow_html=True)

    thresholds = np.linspace(0.01, 0.99, 100)
    savings_list = []
    recall_list  = []
    prec_list    = []

    for thr in thresholds:
        yp = (y_prob >= thr).astype(int)
        tp_t = int(((yp == 1) & (y_test == 1)).sum())
        fp_t = int(((yp == 1) & (y_test == 0)).sum())
        fn_t = int(((yp == 0) & (y_test == 1)).sum())
        saved   = tp_t * avg_fraud_amt
        cost    = fp_t * avg_legit_amt * 0.02
        net     = saved - cost
        r_val   = tp_t / (tp_t + fn_t) if (tp_t + fn_t) > 0 else 0
        p_val   = tp_t / (tp_t + fp_t) if (tp_t + fp_t) > 0 else 0
        savings_list.append(net)
        recall_list.append(r_val)
        prec_list.append(p_val)

    fig_sweep = make_subplots(specs=[[{"secondary_y": True}]])
    fig_sweep.add_trace(go.Scatter(
        x=thresholds, y=savings_list,
        name="Net Savings ($)", line=dict(color=LEGIT_COLOR, width=2.5),
        fill="tozeroy", fillcolor="rgba(0,184,148,0.1)",
    ), secondary_y=False)
    fig_sweep.add_trace(go.Scatter(
        x=thresholds, y=recall_list,
        name="Recall", line=dict(color=FRAUD_COLOR, width=2, dash="dot"),
    ), secondary_y=True)
    fig_sweep.add_trace(go.Scatter(
        x=thresholds, y=prec_list,
        name="Precision", line=dict(color=BLUE_COLOR, width=2, dash="dash"),
    ), secondary_y=True)
    # Mark current threshold
    fig_sweep.add_vline(
        x=threshold, line_color="#FDCB6E", line_width=2, line_dash="dot",
        annotation_text=f"Threshold={threshold:.2f}",
        annotation_font_color="#FDCB6E",
    )
    fig_sweep.update_layout(
        title="Net Financial Savings & Metrics vs Decision Threshold",
        xaxis_title="Threshold",
        legend=dict(orientation="h", y=1.1),
        **{
            "yaxis":  dict(title="Net Savings ($)", color=LEGIT_COLOR),
            "yaxis2": dict(title="Score (0–1)", color=FRAUD_COLOR, overlaying="y", side="right"),
        },
    )
    fig_sweep.update_layout(
        template=PLOTLY_TEMPLATE,
        paper_bgcolor=CARD_BG,
        plot_bgcolor=CARD_BG,
        font=dict(family="Inter", color="#C0C0D8"),
        height=420,
        margin=dict(l=20, r=20, t=40, b=20),
    )
    st.plotly_chart(fig_sweep, use_container_width=True)

    # ── Confusion Matrix at current threshold ─────────────────────────────────
    st.markdown("---")
    st.markdown(f"<div class='section-header'>Confusion Matrix @ Threshold {threshold:.2f}</div>", unsafe_allow_html=True)

    cm_thresh = confusion_matrix(y_test, y_pred_thresh)
    labels = ["Legit (0)", "Fraud (1)"]
    fig_cm_t = go.Figure(go.Heatmap(
        z=cm_thresh,
        x=labels, y=labels,
        colorscale=[[0, CARD_BG], [1, FRAUD_COLOR]],
        text=[[str(v) for v in row] for row in cm_thresh],
        texttemplate="<b>%{text}</b>",
        textfont=dict(size=18),
        showscale=False,
    ))
    fig_cm_t.update_layout(title=f"Confusion Matrix — {best_name} @ {threshold:.2f}", xaxis_title="Predicted", yaxis_title="Actual")
    col_cm, col_note = st.columns([1, 1])
    with col_cm:
        st.plotly_chart(plotly_defaults(fig_cm_t, 340), use_container_width=True)
    with col_note:
        st.markdown(f"""
        <div style='background:#1A1A2E;border:1px solid #2D2D44;border-radius:12px;padding:1.5rem;margin-top:0.5rem;'>
          <p style='color:#FDCB6E;font-weight:700;font-size:1.05rem;'>📊 Cost-Benefit Summary</p>
          <table style='width:100%;color:#C0C0D8;font-size:0.9rem;'>
            <tr><td>✅ Fraud caught (TP)</td><td style='text-align:right;color:#00B894;font-weight:600;'>{tp:,}</td></tr>
            <tr><td>❌ Missed fraud (FN)</td><td style='text-align:right;color:#E94560;font-weight:600;'>{fn:,}</td></tr>
            <tr><td>⚠️ False alarms (FP)</td><td style='text-align:right;color:#FDCB6E;font-weight:600;'>{fp:,}</td></tr>
            <tr><td>✔️ Correct legit (TN)</td><td style='text-align:right;color:#74B9FF;font-weight:600;'>{tn:,}</td></tr>
            <tr><td colspan='2'><hr style='border-color:#2D2D44;'></td></tr>
            <tr><td>💰 Fraud value saved</td><td style='text-align:right;color:#00B894;font-weight:600;'>${fraud_saved:,.0f}</td></tr>
            <tr><td>⚡ False alarm cost</td><td style='text-align:right;color:#FDCB6E;font-weight:600;'>${false_alarm_cost:,.0f}</td></tr>
            <tr><td><b>🏆 Net Benefit</b></td><td style='text-align:right;color:{"#00B894" if net_savings>0 else "#E94560"};font-weight:700;font-size:1.05rem;'>${net_savings:,.0f}</td></tr>
          </table>
        </div>
        """, unsafe_allow_html=True)


# ═══════════════════════════════════════════════════════════════════════════════
# SIDEBAR & MAIN ENTRY POINT
# ═══════════════════════════════════════════════════════════════════════════════
def main():
    # ── Sidebar ───────────────────────────────────────────────────────────────
    with st.sidebar:
        st.markdown("""
        <div style='text-align:center;padding:1rem 0 0.5rem 0;'>
          <div style='font-size:2.5rem;'>🛡️</div>
          <div style='font-size:1.05rem;font-weight:800;color:#FFFFFF;letter-spacing:0.02em;'>FraudShield</div>
          <div style='font-size:0.72rem;color:#8888AA;letter-spacing:0.08em;text-transform:uppercase;'>
            Credit Card Fraud Detection
          </div>
        </div>
        <hr style='border-color:#2D2D44;margin:0.75rem 0;'/>
        """, unsafe_allow_html=True)

        page = st.radio(
            "Navigation",
            options=[
                "📊 Overview",
                "🔍 EDA & Analysis",
                "🤖 ML Models",
                "🎯 Fraud Predictor",
                "💼 Business Impact",
            ],
            label_visibility="collapsed",
        )

        st.markdown("<hr style='border-color:#2D2D44;margin:1rem 0;'/>", unsafe_allow_html=True)
        st.markdown("""
        <div style='font-size:0.75rem;color:#555577;text-align:center;'>
          Dataset: creditcard.csv<br>
          284,807 transactions · 31 features<br>
          <span style='color:#E94560;'>SMOTE + Random Forest / XGBoost / LR</span>
        </div>
        """, unsafe_allow_html=True)

    # ── Load Data ─────────────────────────────────────────────────────────────
    df = load_data()

    if df is None:
        st.error(
            "⚠️ **Dataset not found!**\n\n"
            f"Please place `creditcard.csv` in `{os.path.abspath('.')}` "
            "or set the `DATASET_PATH` environment variable in your `.env` file."
        )
        st.stop()

    # ── Route to Pages ────────────────────────────────────────────────────────
    if page == "📊 Overview":
        page_overview(df)

    elif page == "🔍 EDA & Analysis":
        page_eda(df)

    elif page in ("🤖 ML Models", "🎯 Fraud Predictor", "💼 Business Impact"):
        # Training is deferred until one of these pages is visited
        df_hash = str(len(df))  # Cheap deterministic cache key
        with st.spinner("🔄 Training models… (cached after first run, may take 1-2 min)"):
            results, best_name, explainer, scaler, X_test, y_test, feature_cols = train_models(df_hash)

        if page == "🤖 ML Models":
            page_ml_models(results, best_name, feature_cols)
        elif page == "🎯 Fraud Predictor":
            page_predictor(results, best_name, explainer, scaler, feature_cols)
        elif page == "💼 Business Impact":
            page_business_impact(df, results, best_name)


if __name__ == "__main__":
    main()

