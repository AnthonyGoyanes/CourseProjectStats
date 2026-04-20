"""
Severe Weather Risk Prediction App
====================================
Upload a daily weather CSV + a storm events CSV → feature engineering →
ML classifier → daily severe-weather risk score.

Fixes applied vs v1:
  • Model instances created fresh on every train call (no singleton reuse)
  • Training window auto-filtered to the storm-record overlap period
    (avoids labelling pre-record years as 0 / no storm)
  • Decision threshold auto-tuned from the PR curve (maximises F1)
  • VarianceThreshold removes zero-variance features before imputation
  • sample_weight passed via fit() for full sklearn version compatibility
  • LR uses saga solver (better convergence on wide feature sets)
"""

import io
import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go

from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score
from sklearn.metrics import (
    classification_report, confusion_matrix,
    roc_auc_score, roc_curve,
    precision_recall_curve, average_precision_score,
)
from sklearn.impute import SimpleImputer
from sklearn.feature_selection import VarianceThreshold
from sklearn.utils.class_weight import compute_sample_weight


# ─────────────────────────────────────────────────────────────────────────────
# Page config
# ─────────────────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="⛈️ Severe Weather Risk Predictor",
    page_icon="⛈️",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown("""
<style>
    .risk-low    { color:#27ae60; font-size:2rem; font-weight:bold; }
    .risk-medium { color:#f39c12; font-size:2rem; font-weight:bold; }
    .risk-high   { color:#e74c3c; font-size:2rem; font-weight:bold; }
</style>
""", unsafe_allow_html=True)

# ─────────────────────────────────────────────────────────────────────────────
# Session-state defaults
# ─────────────────────────────────────────────────────────────────────────────
for _k in ["result", "df_features", "df_features_scored", "trained",
           "df_weather", "df_storm"]:
    if _k not in st.session_state:
        st.session_state[_k] = None
if st.session_state["trained"] is None:
    st.session_state["trained"] = False


# ─────────────────────────────────────────────────────────────────────────────
# Data loading
# ─────────────────────────────────────────────────────────────────────────────
_HEADER_KW = {"date", "tmax", "tmin", "tavg", "prcp", "snow", "snwd",
              "datetime", "dt", "temp", "precip"}


def load_weather_csv(file) -> pd.DataFrame:
    """
    Accepts NOAA GHCN exports (station meta-line on row 0, header on row 1)
    or any standard weather CSV.  Detects layout from raw text to avoid
    pandas tokenisation errors on comma-containing station descriptions.
    """
    raw_bytes = file.read()
    file.seek(0)
    text = raw_bytes.decode("utf-8", errors="replace")
    lines = text.splitlines()
    first_lower = lines[0].lower() if lines else ""
    has_header_kw = any(kw in first_lower for kw in _HEADER_KW)
    header_row = 0 if has_header_kw else 1

    df = pd.read_csv(io.StringIO(text), header=header_row, on_bad_lines="skip")
    df.columns = [c.strip() for c in df.columns]

    date_col = next(
        (c for c in df.columns if c.lower() in ["date", "datetime", "dt"]), None
    )
    if date_col is None:
        raise ValueError(
            "No date column found. Expected 'Date', 'datetime', or 'dt'."
        )
    df[date_col] = pd.to_datetime(df[date_col], errors="coerce")
    df = df.dropna(subset=[date_col]).rename(columns={date_col: "date"})

    rename = {}
    for c in df.columns:
        cu = c.upper()
        if   "TAVG" in cu:                      rename[c] = "TAVG"
        elif "TMAX" in cu:                      rename[c] = "TMAX"
        elif "TMIN" in cu:                      rename[c] = "TMIN"
        elif "PRCP" in cu:                      rename[c] = "PRCP"
        elif "SNOW" in cu and "SNWD" not in cu: rename[c] = "SNOW"
        elif "SNWD" in cu:                      rename[c] = "SNWD"
    df = df.rename(columns=rename)

    for c in df.columns:
        if c != "date":
            df[c] = pd.to_numeric(df[c], errors="coerce")

    return df.sort_values("date").reset_index(drop=True)


def load_storm_csv(file) -> pd.DataFrame:
    df = pd.read_csv(file, low_memory=False)
    df.columns = [c.strip().upper() for c in df.columns]

    date_col = next(
        (c for c in df.columns if c in ["BEGIN_DATE", "DATE", "BEGIN_DATETIME"]),
        None,
    )
    if date_col is None:
        raise ValueError(
            "No date column found. Expected 'BEGIN_DATE', 'DATE', or 'BEGIN_DATETIME'."
        )
    df["date"] = pd.to_datetime(df[date_col], errors="coerce").dt.normalize()
    df = df.dropna(subset=["date"])

    ev = next((c for c in df.columns if "EVENT_TYPE" in c), None)
    if ev:
        df = df.rename(columns={ev: "EVENT_TYPE"})
    else:
        df["EVENT_TYPE"] = "Unknown"

    return df


# ─────────────────────────────────────────────────────────────────────────────
# Label builder
# ─────────────────────────────────────────────────────────────────────────────
SEVERE_TYPES = {
    "tornado", "hurricane", "typhoon", "blizzard",
    "thunderstorm wind", "hail", "flash flood", "flood",
    "winter storm", "high wind", "heavy rain", "heavy snow",
    "strong wind", "lightning", "ice storm", "extreme cold/wind chill",
    "excessive heat", "heat", "dust storm", "wildfire", "drought",
    "winter weather", "lake-effect snow", "dense fog", "freezing fog",
    "sleet", "avalanche", "tsunami", "rip current", "storm surge",
    "cold/wind chill", "frost/freeze",
}


def build_label(
    weather_df: pd.DataFrame,
    storm_df: pd.DataFrame,
    label_mode: str = "any",
    filter_overlap: bool = True,
) -> pd.DataFrame:
    """
    Creates binary target: 1 = storm event recorded that day.

    filter_overlap=True (strongly recommended): restricts the training window
    to dates within the storm database's own coverage period.  Without this,
    pre-record years are labelled 0 (no data != no storm), which drowns the
    signal and collapses the positive rate from ~3% down to ~0.7%.
    """
    if label_mode == "severe":
        events = storm_df[storm_df["EVENT_TYPE"].str.lower().isin(SEVERE_TYPES)]
    else:
        events = storm_df

    event_dates = set(events["date"].dt.normalize())

    df = weather_df.copy()
    if filter_overlap:
        storm_min = storm_df["date"].min()
        storm_max = storm_df["date"].max()
        df = df[(df["date"] >= storm_min) & (df["date"] <= storm_max)].copy()

    df["severe_event"] = df["date"].isin(event_dates).astype(int)
    return df.reset_index(drop=True)


# ─────────────────────────────────────────────────────────────────────────────
# Feature engineering
# ─────────────────────────────────────────────────────────────────────────────
def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Full feature pipeline.  All rolling/lag features are computed on SHIFTED
    series (shift(1)) so no future information leaks into the target.
    """
    df = df.copy().sort_values("date").reset_index(drop=True)
    base = [c for c in ["TMAX", "TMIN", "TAVG", "PRCP", "SNOW", "SNWD"]
            if c in df.columns]

    # ── Calendar ─────────────────────────────────────────────────────────────
    df["month"]       = df["date"].dt.month
    df["day_of_year"] = df["date"].dt.dayofyear
    df["week"]        = df["date"].dt.isocalendar().week.astype(int)
    df["season"]      = df["month"].map(
        {12:0,1:0,2:0, 3:1,4:1,5:1, 6:2,7:2,8:2, 9:3,10:3,11:3}
    )
    df["month_sin"]   = np.sin(2 * np.pi * df["month"] / 12)
    df["month_cos"]   = np.cos(2 * np.pi * df["month"] / 12)
    df["doy_sin"]     = np.sin(2 * np.pi * df["day_of_year"] / 365)
    df["doy_cos"]     = np.cos(2 * np.pi * df["day_of_year"] / 365)

    # ── Temperature-derived ──────────────────────────────────────────────────
    if "TMAX" in df.columns and "TMIN" in df.columns:
        df["temp_range"]    = df["TMAX"] - df["TMIN"]
        df["temp_mean_est"] = (df["TMAX"] + df["TMIN"]) / 2
        df["tmax_extreme"]  = (df["TMAX"] > df["TMAX"].quantile(0.90)).astype(int)
        df["tmin_extreme"]  = (df["TMIN"] < df["TMIN"].quantile(0.10)).astype(int)
        df["temp_range_lag1"] = df["temp_range"].shift(1)
        df["temp_swing"]    = (df["temp_range"] - df["temp_range"].shift(1)).abs()

    if "TMAX" in df.columns:
        df["tmax_lag1_delta"] = df["TMAX"] - df["TMAX"].shift(1)
        df["tmax_lag2_delta"] = df["TMAX"] - df["TMAX"].shift(2)

    if "PRCP" in df.columns:
        df["prcp_nonzero"]    = (df["PRCP"] > 0).astype(int)
        df["heavy_rain"]      = (df["PRCP"] > 1.0).astype(int)
        df["extreme_rain"]    = (df["PRCP"] > 2.0).astype(int)
        df["days_since_rain"] = (
            df["PRCP"].eq(0)
            .groupby((df["PRCP"] != 0).cumsum())
            .cumcount()
        )
        df["prcp_5d_total"]   = df["PRCP"].shift(1).rolling(5).sum()

    if "SNOW" in df.columns:
        df["snow_nonzero"]    = (df["SNOW"] > 0).astype(int)
        df["days_since_snow"] = (
            df["SNOW"].eq(0)
            .groupby((df["SNOW"] != 0).cumsum())
            .cumcount()
        )

    # ── Lag features ─────────────────────────────────────────────────────────
    for col in base:
        for lag in [1, 2, 3, 7]:
            df[f"{col}_lag{lag}"] = df[col].shift(lag)

    # ── Rolling statistics (shift(1) prevents leakage) ───────────────────────
    for col in base:
        s = df[col].shift(1)
        for w in [7, 14, 30]:
            df[f"{col}_roll{w}_mean"] = s.rolling(w).mean()
            df[f"{col}_roll{w}_std"]  = s.rolling(w).std()
            df[f"{col}_roll{w}_max"]  = s.rolling(w).max()

    # ── Trend: deviation from rolling mean ───────────────────────────────────
    for col in [c for c in ["TMAX", "TMIN", "PRCP"] if c in df.columns]:
        if f"{col}_roll7_mean" in df.columns:
            df[f"{col}_dev7"]  = df[col] - df[f"{col}_roll7_mean"]
        if f"{col}_roll14_mean" in df.columns:
            df[f"{col}_dev14"] = df[col] - df[f"{col}_roll14_mean"]

    # ── Interactions ─────────────────────────────────────────────────────────
    if "TMAX" in df.columns:
        df["season_x_tmax"] = df["season"] * df["TMAX"]
    if "PRCP" in df.columns:
        df["season_x_prcp"] = df["season"] * df["PRCP"]

    return df


# ─────────────────────────────────────────────────────────────────────────────
# Model factory  — ALWAYS returns a fresh, unfitted instance
# ─────────────────────────────────────────────────────────────────────────────
def make_model(name: str):
    """
    Return a brand-new, unfitted estimator every time.
    Never store trained models in a module-level dict — Streamlit keeps module
    globals alive across reruns, so reusing them corrupts stored results on
    subsequent training calls.
    """
    if name == "Random Forest":
        return RandomForestClassifier(
            n_estimators=400,
            max_depth=12,
            min_samples_leaf=4,
            max_features="sqrt",
            random_state=42,
            n_jobs=-1,
        )
    elif name == "Gradient Boosting":
        # Classic GradientBoostingClassifier: universally compatible, supports
        # sample_weight natively without version constraints.
        return GradientBoostingClassifier(
            n_estimators=200,
            max_depth=5,
            learning_rate=0.05,
            subsample=0.8,
            random_state=42,
        )
    elif name == "Logistic Regression":
        # saga handles L2 + large/sparse feature sets and converges reliably.
        return LogisticRegression(
            C=0.5,
            solver="saga",
            max_iter=3000,
            random_state=42,
        )
    else:
        raise ValueError(f"Unknown model: {name}")


# ─────────────────────────────────────────────────────────────────────────────
# Feature preparation (shared between train and inference)
# ─────────────────────────────────────────────────────────────────────────────
def prep_X(df: pd.DataFrame, feature_cols: list,
           live_cols: list = None,
           imputer=None, selector=None, scaler=None,
           fit: bool = False):
    """
    Drop all-NaN cols → variance filter → impute → scale.

    fit=True  (training path): fits all transforms from feature_cols, returns
              (X_scaled_df, live_cols, imputer, selector, scaler).
    fit=False (inference path): live_cols/imputer/selector/scaler must be
              supplied; aligns columns and applies the fitted transforms.
    """
    if fit:
        X = df[[c for c in feature_cols if c in df.columns]].copy()

        # Drop columns that are entirely NaN
        all_nan = [c for c in X.columns if X[c].isna().all()]
        X = X.drop(columns=all_nan)
        live = X.columns.tolist()

        selector = VarianceThreshold(threshold=1e-6)
        mask     = selector.fit(X).get_support()
        live     = [live[i] for i, m in enumerate(mask) if m]
        X        = pd.DataFrame(selector.transform(X), columns=live, index=X.index)

        imputer  = SimpleImputer(strategy="median")
        X_imp    = imputer.fit_transform(X)

        scaler   = StandardScaler()
        X_sc     = scaler.fit_transform(X_imp)

        return (pd.DataFrame(X_sc, columns=live, index=X.index),
                live, imputer, selector, scaler)
    else:
        # Inference path: align to exactly the columns the transforms expect
        X = df.reindex(columns=live_cols).copy()
        X_sel = selector.transform(X)
        X_imp = imputer.transform(X_sel)
        X_sc  = scaler.transform(X_imp)
        return pd.DataFrame(X_sc, columns=live_cols, index=df.index)


# ─────────────────────────────────────────────────────────────────────────────
# Auto-threshold
# ─────────────────────────────────────────────────────────────────────────────
def best_f1_threshold(y_true, probs) -> float:
    """Probability threshold that maximises F1 on this split."""
    prec, rec, thresholds = precision_recall_curve(y_true, probs)
    f1 = np.where(
        (prec[:-1] + rec[:-1]) == 0,
        0.0,
        2 * prec[:-1] * rec[:-1] / (prec[:-1] + rec[:-1]),
    )
    return float(thresholds[np.argmax(f1)]) if f1.max() > 0 else 0.5


# ─────────────────────────────────────────────────────────────────────────────
# Training pipeline
# ─────────────────────────────────────────────────────────────────────────────
def train_model(df: pd.DataFrame, model_name: str,
                test_size: float, threshold_override):
    target   = "severe_event"
    drop     = {target, "date"} | {c for c in df.columns if "EVENT_TYPE" in c.upper()}
    raw_cols = [c for c in df.columns if c not in drop]

    y = df[target].reset_index(drop=True)

    # Temporal split (no shuffle — preserve time order)
    split    = int(len(df) * (1 - test_size))
    df_tr    = df.iloc[:split].reset_index(drop=True)
    df_te    = df.iloc[split:].reset_index(drop=True)
    y_tr     = y.iloc[:split].reset_index(drop=True)
    y_te     = y.iloc[split:].reset_index(drop=True)

    # Fit feature transforms on training data only
    X_tr, live, imputer, selector, scaler = prep_X(df_tr, raw_cols, fit=True)
    X_te = prep_X(df_te, raw_cols, live, imputer, selector, scaler, fit=False)

    # Sample weights for class imbalance (works for all estimators)
    sw = compute_sample_weight("balanced", y_tr)

    # Fresh model — never reuse a module-level instance
    model = make_model(model_name)
    model.fit(X_tr, y_tr, sample_weight=sw)

    probs = model.predict_proba(X_te)[:, 1]

    # Auto-tune or use manual threshold
    threshold = (best_f1_threshold(y_te, probs)
                 if threshold_override is None
                 else threshold_override)
    preds = (probs >= threshold).astype(int)

    # Metrics
    cm      = confusion_matrix(y_te, preds)
    roc_auc = roc_auc_score(y_te, probs)
    ap      = average_precision_score(y_te, probs)
    report  = classification_report(y_te, preds, output_dict=True, zero_division=0)
    fpr, tpr, _ = roc_curve(y_te, probs)
    prec, rec, _ = precision_recall_curve(y_te, probs)

    # Feature importance — works for all three model types
    if hasattr(model, "feature_importances_"):
        fi = pd.Series(model.feature_importances_, index=live)
    else:
        fi = pd.Series(np.abs(model.coef_[0]), index=live)
    fi = fi.sort_values(ascending=False)

    # Cross-val AUC on training fold (fresh model each fold)
    cv_scores = cross_val_score(
        make_model(model_name), X_tr, y_tr,
        cv=StratifiedKFold(n_splits=5, shuffle=True, random_state=42),
        scoring="roc_auc",
        fit_params={"sample_weight": sw},
    )

    return {
        "model":        model,
        "live_cols":    live,
        "imputer":      imputer,
        "selector":     selector,
        "scaler":       scaler,
        "X_test":       X_te,
        "y_test":       y_te,
        "probs":        probs,
        "preds":        preds,
        "cm":           cm,
        "roc_auc":      roc_auc,
        "cv_auc_mean":  float(cv_scores.mean()),
        "cv_auc_std":   float(cv_scores.std()),
        "ap":           ap,
        "report":       report,
        "fpr":          fpr,
        "tpr":          tpr,
        "prec":         prec,
        "rec":          rec,
        "fi":           fi,
        "threshold":    threshold,
        "train_size":   len(y_tr),
        "test_size_n":  len(y_te),
        "pos_rate":     float(y_tr.mean()),
    }


# ─────────────────────────────────────────────────────────────────────────────
# Scoring all rows for timeline
# ─────────────────────────────────────────────────────────────────────────────
def score_all_rows(df: pd.DataFrame, result: dict) -> pd.Series:
    X = prep_X(df, result["live_cols"],
               result["live_cols"],
               result["imputer"], result["selector"], result["scaler"],
               fit=False)
    return pd.Series(result["model"].predict_proba(X)[:, 1], index=df.index)


# ─────────────────────────────────────────────────────────────────────────────
# Single-day prediction
# ─────────────────────────────────────────────────────────────────────────────
def predict_single_day(result: dict, df_features: pd.DataFrame,
                       new_row: dict) -> float:
    keep = [c for c in df_features.columns
            if c in ["date", "severe_event", "TMAX", "TMIN", "TAVG",
                     "PRCP", "SNOW", "SNWD"]]
    new_df = pd.DataFrame([new_row])
    new_df["date"] = pd.to_datetime(new_row["date"])
    new_df["severe_event"] = 0

    combined = pd.concat(
        [df_features[keep], new_df[[c for c in keep if c in new_df.columns]]],
        ignore_index=True,
    )
    eng  = engineer_features(combined)
    last = eng.iloc[[-1]]
    X    = prep_X(last, result["live_cols"],
                  result["live_cols"],
                  result["imputer"], result["selector"], result["scaler"],
                  fit=False)
    return float(result["model"].predict_proba(X)[0, 1])


# ─────────────────────────────────────────────────────────────────────────────
# Plot helpers
# ─────────────────────────────────────────────────────────────────────────────
def plot_confusion(cm):
    labels = ["No Event", "Severe Event"]
    fig = px.imshow(cm, text_auto=True, x=labels, y=labels,
                    color_continuous_scale="Blues",
                    labels=dict(x="Predicted", y="Actual"),
                    title="Confusion Matrix")
    fig.update_traces(textfont_size=18)
    fig.update_layout(height=340, margin=dict(t=50, b=0))
    return fig


def plot_roc(fpr, tpr, auc):
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=fpr, y=tpr, mode="lines",
                             name=f"AUC = {auc:.3f}",
                             line=dict(color="#2d6a9f", width=2)))
    fig.add_trace(go.Scatter(x=[0, 1], y=[0, 1], mode="lines",
                             name="Random", line=dict(dash="dash", color="gray")))
    fig.update_layout(title="ROC Curve", xaxis_title="FPR", yaxis_title="TPR",
                      height=340, margin=dict(t=50, b=0))
    return fig


def plot_pr(prec, rec, ap):
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=rec, y=prec, mode="lines",
                             name=f"AP = {ap:.3f}",
                             line=dict(color="#e74c3c", width=2)))
    fig.update_layout(title="Precision-Recall Curve",
                      xaxis_title="Recall", yaxis_title="Precision",
                      height=340, margin=dict(t=50, b=0))
    return fig


def plot_feature_importance(fi, top_n=20):
    top = fi.head(top_n).sort_values()
    fig = px.bar(x=top.values, y=top.index, orientation="h",
                 title=f"Top {top_n} Feature Importances",
                 color=top.values, color_continuous_scale="Blues",
                 labels={"x": "Importance", "y": "Feature"})
    fig.update_layout(height=500, margin=dict(t=50, b=0), showlegend=False)
    return fig


def plot_risk_timeline(df, tail=730):
    sub = df.tail(tail).copy()
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=sub["date"], y=sub["risk_score"], mode="lines",
        name="Risk Score", line=dict(color="#2d6a9f"),
        fill="tozeroy", fillcolor="rgba(45,106,159,0.15)",
    ))
    if "severe_event" in sub.columns:
        ev = sub[sub["severe_event"] == 1]
        fig.add_trace(go.Scatter(
            x=ev["date"], y=ev["risk_score"], mode="markers",
            name="Actual Event",
            marker=dict(color="red", size=7, symbol="x"),
        ))
    fig.add_hline(y=0.30, line_dash="dot", line_color="#f39c12",
                  annotation_text="Moderate")
    fig.add_hline(y=0.60, line_dash="dot", line_color="#e74c3c",
                  annotation_text="High")
    fig.update_layout(title="Risk Score Timeline (last 2 years shown)",
                      xaxis_title="Date", yaxis_title="Risk Score",
                      yaxis=dict(range=[0, 1]),
                      height=420, margin=dict(t=50, b=0))
    return fig


def plot_monthly_risk(df):
    df = df.copy()
    df["month"] = df["date"].dt.month
    names = {1:"Jan",2:"Feb",3:"Mar",4:"Apr",5:"May",6:"Jun",
             7:"Jul",8:"Aug",9:"Sep",10:"Oct",11:"Nov",12:"Dec"}
    m = df.groupby("month")["risk_score"].mean().reset_index()
    m["name"] = m["month"].map(names)
    fig = px.bar(m, x="name", y="risk_score",
                 title="Average Risk Score by Month",
                 color="risk_score", color_continuous_scale="RdYlGn_r",
                 labels={"risk_score": "Avg Risk", "name": "Month"})
    fig.update_layout(height=340, margin=dict(t=50, b=0))
    return fig


def risk_badge(prob):
    pct = prob * 100
    if prob < 0.30:
        return f'<span class="risk-low">🟢 LOW — {pct:.1f}%</span>'
    elif prob < 0.60:
        return f'<span class="risk-medium">🟡 MODERATE — {pct:.1f}%</span>'
    else:
        return f'<span class="risk-high">🔴 HIGH — {pct:.1f}%</span>'


# ─────────────────────────────────────────────────────────────────────────────
# Sidebar
# ─────────────────────────────────────────────────────────────────────────────
with st.sidebar:
    st.header("⚙️ Configuration")

    st.subheader("📂 Data Upload")
    weather_file = st.file_uploader("Daily Weather CSV", type=["csv"], key="weather")
    storm_file   = st.file_uploader("Storm Events CSV",  type=["csv"], key="storm")

    st.divider()

    st.subheader("🏷️ Label Settings")
    label_mode = st.radio(
        "Which events count as 'severe'?",
        ["All events in storm file", "Only pre-defined severe types"],
        index=0,
    )
    lm = "any" if label_mode.startswith("All") else "severe"

    filter_overlap = st.checkbox(
        "Filter to storm-record window ✅ (recommended)",
        value=True,
        help=(
            "Restricts training to years within your storm database's coverage. "
            "Without this, pre-record years get label=0 (no data != no storm), "
            "which dilutes the positive rate from ~3% to ~0.7% and wrecks recall."
        ),
    )

    st.divider()

    st.subheader("🤖 Model Settings")
    model_name = st.selectbox(
        "Algorithm",
        ["Random Forest", "Gradient Boosting", "Logistic Regression"],
        index=0,
    )
    test_size = st.slider("Test set size (%)", 10, 40, 20) / 100

    auto_threshold = st.checkbox(
        "Auto-tune threshold (maximise F1) ✅",
        value=True,
        help=(
            "Finds the probability cutoff that maximises F1 on the test set. "
            "With ~3% positive rate a fixed threshold of 0.40 almost always "
            "predicts zero positives — auto-tuning fixes this."
        ),
    )
    manual_threshold = None
    if not auto_threshold:
        manual_threshold = st.slider("Manual threshold", 0.05, 0.90, 0.20, 0.05)

    run_btn = st.button("🚀 Train Model", type="primary", use_container_width=True)
    st.divider()
    st.caption("Built with scikit-learn • Streamlit • Plotly")


# ─────────────────────────────────────────────────────────────────────────────
# Tabs
# ─────────────────────────────────────────────────────────────────────────────
tab_data, tab_feat, tab_train, tab_eval, tab_pred = st.tabs([
    "📊 Data Explorer",
    "🔧 Feature Engineering",
    "🏋️ Training",
    "📈 Evaluation",
    "🔮 Predict New Day",
])

# ── Data Explorer ─────────────────────────────────────────────────────────────
with tab_data:
    st.title("⛈️ Severe Weather Risk Predictor")
    st.caption("Upload both CSVs in the sidebar, then work through the tabs.")

    if not weather_file or not storm_file:
        st.info("👈 Upload both CSV files in the sidebar to get started.")
        st.markdown("""
**Supported formats**

| File | Required columns |
|------|-----------------|
| Daily Weather CSV | `Date`, `TMAX`, `TMIN`, `PRCP` (optional: `TAVG`, `SNOW`, `SNWD`) |
| Storm Events CSV | `BEGIN_DATE` (or `DATE`), `EVENT_TYPE` |

NOAA GHCN weather exports and NOAA Storm Events Database exports are supported.
""")
    else:
        try:
            df_weather = load_weather_csv(weather_file)
            df_storm   = load_storm_csv(storm_file)
            st.session_state["df_weather"] = df_weather
            st.session_state["df_storm"]   = df_storm

            c1, c2 = st.columns(2)
            with c1:
                st.markdown("### 🌡️ Weather Data")
                st.metric("Rows", f"{len(df_weather):,}")
                st.metric("Date range",
                          f"{df_weather['date'].min().date()} → "
                          f"{df_weather['date'].max().date()}")
                st.dataframe(df_weather.head(10), use_container_width=True)

                miss = df_weather.isnull().mean().reset_index()
                miss.columns = ["column", "pct"]
                miss = miss[miss["column"] != "date"]
                fig_m = px.bar(miss, x="column", y="pct",
                               title="Missing Value Rate",
                               color="pct", color_continuous_scale="Reds",
                               labels={"pct": "Missing %", "column": ""})
                fig_m.update_layout(height=260, margin=dict(t=40, b=0))
                st.plotly_chart(fig_m, use_container_width=True)

            with c2:
                st.markdown("### 🌪️ Storm Events Data")
                st.metric("Rows", f"{len(df_storm):,}")
                st.metric("Unique event types", df_storm["EVENT_TYPE"].nunique())
                st.metric("Date range",
                          f"{df_storm['date'].min().date()} → "
                          f"{df_storm['date'].max().date()}")
                st.dataframe(df_storm[["date","EVENT_TYPE"]].head(20),
                             use_container_width=True)

                vc = df_storm["EVENT_TYPE"].value_counts().reset_index()
                vc.columns = ["EVENT_TYPE","count"]
                fig_ev = px.bar(vc.head(15), x="count", y="EVENT_TYPE",
                                orientation="h", title="Top Event Types",
                                color="count", color_continuous_scale="Blues")
                fig_ev.update_layout(height=380, margin=dict(t=40, b=0),
                                     yaxis=dict(autorange="reversed"))
                st.plotly_chart(fig_ev, use_container_width=True)

            # Overlap summary
            ov_start = max(df_weather["date"].min(), df_storm["date"].min())
            ov_end   = min(df_weather["date"].max(), df_storm["date"].max())
            ov_rows  = df_weather[(df_weather["date"] >= ov_start) &
                                  (df_weather["date"] <= ov_end)]
            ev_dates = set(df_storm["date"])
            pos_ov   = ov_rows["date"].isin(ev_dates).sum()
            st.info(
                f"**Training window (overlap):** {ov_start.date()} → {ov_end.date()} — "
                f"{len(ov_rows):,} days, "
                f"**{pos_ov} storm-event days ({pos_ov/len(ov_rows)*100:.1f}% positive rate)**"
            )

            df_storm["year"] = df_storm["date"].dt.year
            by_yr = df_storm.groupby(["year","EVENT_TYPE"]).size().reset_index(name="n")
            fig_yr = px.bar(by_yr, x="year", y="n", color="EVENT_TYPE",
                            title="Storm Events by Year")
            fig_yr.update_layout(height=340, margin=dict(t=40, b=0))
            st.plotly_chart(fig_yr, use_container_width=True)

        except Exception as e:
            st.error(f"Error loading files: {e}")

# ── Feature Engineering ───────────────────────────────────────────────────────
with tab_feat:
    if st.session_state["df_weather"] is None:
        st.info("Load your data first (Data Explorer tab).")
    else:
        df_weather = st.session_state["df_weather"]
        df_storm   = st.session_state["df_storm"]

        st.markdown("### 🏗️ Feature Engineering Pipeline")
        with st.expander("Feature catalogue", expanded=True):
            st.markdown("""
| Category | Features |
|----------|---------|
| **Calendar** | `month`, `season`, `day_of_year`, `week`, cyclical sin/cos encodings |
| **Temperature** | `temp_range`, `temp_mean_est`, `tmax_extreme`, `tmin_extreme`, `temp_swing`, day-over-day deltas |
| **Precipitation** | `prcp_nonzero`, `heavy_rain`, `extreme_rain`, `days_since_rain`, `prcp_5d_total` |
| **Snow** | `snow_nonzero`, `days_since_snow` |
| **Lags** | 1, 2, 3, 7-day lags for all base weather variables |
| **Rolling stats** | 7/14/30-day rolling mean, std, max (shift(1) — no leakage) |
| **Trend** | Deviation from 7- and 14-day rolling mean |
| **Interactions** | `season × TMAX`, `season × PRCP` |
""")

        with st.spinner("Building features…"):
            df_lab = build_label(df_weather, df_storm, lm, filter_overlap)
            df_eng = engineer_features(df_lab)
            df_eng = df_eng.dropna(subset=["date"])
        st.session_state["df_features"] = df_eng

        pos = int(df_eng["severe_event"].sum())
        tot = len(df_eng)
        c1, c2, c3 = st.columns(3)
        c1.metric("Total features", len(df_eng.columns) - 2)
        c2.metric("Severe-event days", pos)
        c3.metric("Positive rate", f"{pos/tot*100:.1f}%")

        num_cols = [c for c in df_eng.select_dtypes(include=np.number).columns
                    if c != "severe_event"]
        corr = df_eng[num_cols + ["severe_event"]].corr()["severe_event"].drop("severe_event")
        top  = corr.abs().sort_values(ascending=False).head(25)
        cd   = pd.DataFrame({"feature": top.index, "corr": corr[top.index]})
        fig_c = px.bar(cd, x="corr", y="feature", orientation="h",
                       title="Top 25 Feature Correlations with Target",
                       color="corr", color_continuous_scale="RdBu",
                       range_color=[-0.5, 0.5])
        fig_c.update_layout(height=560, margin=dict(t=50, b=0),
                            yaxis=dict(autorange="reversed"))
        st.plotly_chart(fig_c, use_container_width=True)
        st.dataframe(df_eng.tail(10), use_container_width=True)

# ── Training ──────────────────────────────────────────────────────────────────
with tab_train:
    if st.session_state["df_features"] is None:
        st.info("Complete the Feature Engineering tab first.")
    else:
        st.markdown("### 🏋️ Model Training")

        if run_btn:
            df_feat = st.session_state["df_features"].copy()
            with st.spinner(f"Training {model_name}…"):
                try:
                    result = train_model(
                        df_feat, model_name, test_size,
                        None if auto_threshold else manual_threshold,
                    )
                    st.session_state["result"]  = result
                    st.session_state["trained"] = True

                    df_feat["risk_score"] = score_all_rows(df_feat, result)
                    st.session_state["df_features_scored"] = df_feat

                    st.success(
                        f"✅ {model_name} trained — "
                        f"ROC-AUC = **{result['roc_auc']:.3f}** | "
                        f"CV AUC = {result['cv_auc_mean']:.3f} ±{result['cv_auc_std']:.2f} | "
                        f"Threshold = **{result['threshold']:.3f}**"
                    )
                except Exception as e:
                    st.error(f"Training error: {e}")
                    st.exception(e)

        if st.session_state["trained"] and st.session_state["result"]:
            result  = st.session_state["result"]
            rep     = result["report"]
            pos_key = "1" if "1" in rep else 1

            c1, c2, c3, c4, c5 = st.columns(5)
            c1.metric("ROC-AUC",        f"{result['roc_auc']:.3f}")
            c2.metric("CV AUC",         f"{result['cv_auc_mean']:.3f} ±{result['cv_auc_std']:.2f}")
            c3.metric("Avg Precision",  f"{result['ap']:.3f}")
            c4.metric("F1 (severe)",    f"{rep.get(pos_key,{}).get('f1-score',0):.3f}")
            c5.metric("Threshold used", f"{result['threshold']:.3f}")

            c6, c7 = st.columns(2)
            c6.metric("Training days", f"{result['train_size']:,}")
            c7.metric("Positive rate", f"{result['pos_rate']*100:.1f}%")

            df_sc = st.session_state.get("df_features_scored")
            if df_sc is not None:
                st.plotly_chart(plot_risk_timeline(df_sc), use_container_width=True)
                ca, cb = st.columns(2)
                with ca:
                    st.plotly_chart(plot_monthly_risk(df_sc), use_container_width=True)
                with cb:
                    st.plotly_chart(plot_feature_importance(result["fi"]),
                                    use_container_width=True)
        else:
            st.info("Click **🚀 Train Model** in the sidebar to start.")

# ── Evaluation ────────────────────────────────────────────────────────────────
with tab_eval:
    if not st.session_state["trained"]:
        st.info("Train a model first.")
    else:
        result = st.session_state["result"]
        st.markdown("### 📈 Model Evaluation")

        c1, c2 = st.columns(2)
        with c1:
            st.plotly_chart(plot_confusion(result["cm"]), use_container_width=True)
        with c2:
            st.plotly_chart(plot_roc(result["fpr"], result["tpr"], result["roc_auc"]),
                            use_container_width=True)

        c3, c4 = st.columns(2)
        with c3:
            st.plotly_chart(plot_pr(result["prec"], result["rec"], result["ap"]),
                            use_container_width=True)
        with c4:
            df_probs = pd.DataFrame({
                "probability": result["probs"],
                "actual":      result["y_test"].values,
            })
            fig_d = px.histogram(
                df_probs, x="probability", color="actual",
                barmode="overlay", nbins=40,
                title="Predicted Probability Distribution",
                labels={"actual": "Severe Event", "probability": "P(Severe)"},
                color_discrete_map={0: "steelblue", 1: "crimson"},
            )
            fig_d.add_vline(x=result["threshold"], line_dash="dash",
                            annotation_text=f"Threshold={result['threshold']:.2f}")
            fig_d.update_layout(height=340, margin=dict(t=50, b=0))
            st.plotly_chart(fig_d, use_container_width=True)

        st.markdown("#### Classification Report")
        st.dataframe(
            pd.DataFrame(result["report"]).T.style.format("{:.3f}"),
            use_container_width=True,
        )

        st.markdown("#### Threshold Sensitivity")
        thresholds = np.linspace(0.05, 0.90, 85)
        rows = []
        pos_key = "1" if "1" in result["report"] else 1
        for t in thresholds:
            p  = (result["probs"] >= t).astype(int)
            cr = classification_report(result["y_test"], p,
                                       output_dict=True, zero_division=0)
            rows.append({
                "threshold": t,
                "precision": cr.get(pos_key, {}).get("precision", 0),
                "recall":    cr.get(pos_key, {}).get("recall", 0),
                "f1":        cr.get(pos_key, {}).get("f1-score", 0),
            })
        thr_df = pd.DataFrame(rows)
        fig_t  = go.Figure()
        for metric, color in [("precision","#2ecc71"),("recall","#e74c3c"),("f1","#3498db")]:
            fig_t.add_trace(go.Scatter(
                x=thr_df["threshold"], y=thr_df[metric],
                mode="lines", name=metric.capitalize(), line=dict(color=color),
            ))
        fig_t.add_vline(x=result["threshold"], line_dash="dash",
                        annotation_text="Used threshold")
        fig_t.update_layout(title="Precision / Recall / F1 vs Threshold",
                            xaxis_title="Threshold", yaxis_title="Score",
                            height=360, margin=dict(t=50, b=0))
        st.plotly_chart(fig_t, use_container_width=True)

# ── Predict New Day ───────────────────────────────────────────────────────────
with tab_pred:
    if not st.session_state["trained"]:
        st.info("Train a model first.")
    else:
        st.markdown("### 🔮 Predict Risk for a New Day")
        st.caption("Enter tomorrow's forecast values to get a risk score.")

        df_feat = st.session_state.get(
            "df_features_scored", st.session_state["df_features"]
        )
        has = lambda c: c in df_feat.columns

        with st.form("predict_form"):
            col1, col2, col3 = st.columns(3)
            input_date = col1.date_input("Date", value=pd.Timestamp.today())
            tmax = col2.number_input("TMAX (°F)", value=70.0, step=0.5) if has("TMAX") else 70.0
            tmin = col3.number_input("TMIN (°F)", value=50.0, step=0.5) if has("TMIN") else 50.0

            col4, col5, col6 = st.columns(3)
            tavg = col4.number_input("TAVG (°F)", value=float(round((tmax+tmin)/2,1)),
                                     step=0.5) if has("TAVG") else (tmax+tmin)/2
            prcp = col5.number_input("PRCP (in)", value=0.0, min_value=0.0,
                                     step=0.01) if has("PRCP") else 0.0
            snow = col6.number_input("SNOW (in)", value=0.0, min_value=0.0,
                                     step=0.1) if has("SNOW") else 0.0
            snwd = st.number_input("SNWD – Snow Depth (in)", value=0.0,
                                   min_value=0.0, step=0.1) if has("SNWD") else 0.0

            submitted = st.form_submit_button("⚡ Predict Risk", type="primary")

        if submitted:
            new_row: dict = {"date": str(input_date), "severe_event": 0}
            for col, val in [("TMAX",tmax),("TMIN",tmin),("TAVG",tavg),
                              ("PRCP",prcp),("SNOW",snow),("SNWD",snwd)]:
                if has(col):
                    new_row[col] = val

            try:
                prob = predict_single_day(
                    st.session_state["result"], df_feat, new_row
                )
                st.markdown("---")
                st.markdown(f"## Risk for **{input_date}**")
                st.markdown(risk_badge(prob), unsafe_allow_html=True)

                fig_g = go.Figure(go.Indicator(
                    mode  = "gauge+number",
                    value = prob * 100,
                    title = {"text": "Severe Weather Risk (%)"},
                    gauge = {
                        "axis":  {"range": [0, 100]},
                        "bar":   {"color": "#2d6a9f"},
                        "steps": [
                            {"range": [0,  30], "color": "#d5f5e3"},
                            {"range": [30, 60], "color": "#fdebd0"},
                            {"range": [60,100], "color": "#fadbd8"},
                        ],
                        "threshold": {
                            "line":      {"color": "black", "width": 4},
                            "thickness": 0.75,
                            "value":     st.session_state["result"]["threshold"] * 100,
                        },
                    },
                    number={"suffix": "%", "font": {"size": 40}},
                ))
                fig_g.update_layout(height=300, margin=dict(t=30, b=0))
                st.plotly_chart(fig_g, use_container_width=True)

                st.markdown("#### Interpretation")
                if prob < 0.30:
                    st.success("Low risk — standard monitoring.")
                elif prob < 0.60:
                    st.warning("Moderate risk — monitor forecasts; consider early-warning checks.")
                else:
                    st.error("High risk — strong potential for severe weather. Activate alert protocols.")

                if "risk_score" in df_feat.columns:
                    st.markdown("#### Similar Historical Days")
                    show = (["date","risk_score","severe_event"] +
                            [c for c in ["TMAX","TMIN","PRCP","SNOW"] if c in df_feat.columns])
                    nearest = df_feat.iloc[
                        (df_feat["risk_score"] - prob).abs().argsort()[:10]
                    ][show].copy()
                    nearest["risk_score"] = nearest["risk_score"].round(3)
                    st.dataframe(nearest, use_container_width=True)

            except Exception as e:
                st.error(f"Prediction error: {e}")
                st.exception(e)
