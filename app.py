"""
Severe Weather Risk Prediction App
===================================
Plug-and-play ML pipeline that ingests:
  1. A daily weather observations CSV  (NOAA GHCN format or similar)
  2. A storm events CSV                 (NOAA Storm Events format or similar)
…and trains a classifier to score the probability of a severe-weather event
on any given day.
"""

import io
import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from sklearn.ensemble import (
    RandomForestClassifier,
    GradientBoostingClassifier,
    HistGradientBoostingClassifier,
)
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    roc_auc_score,
    roc_curve,
    precision_recall_curve,
    average_precision_score,
)
from sklearn.inspection import permutation_importance
from sklearn.impute import SimpleImputer


# ─────────────────────────────────────────────────────────────────────────────
# Page config
# ─────────────────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="⛈️ Severe Weather Risk Predictor",
    page_icon="⛈️",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ─────────────────────────────────────────────────────────────────────────────
# Custom CSS
# ─────────────────────────────────────────────────────────────────────────────
st.markdown("""
<style>
    .metric-card {
        background: linear-gradient(135deg, #1e3a5f 0%, #2d6a9f 100%);
        border-radius: 12px;
        padding: 16px;
        color: white;
        text-align: center;
        box-shadow: 0 4px 15px rgba(0,0,0,0.2);
    }
    .risk-gauge-low    { color: #27ae60; font-size: 2rem; font-weight: bold; }
    .risk-gauge-medium { color: #f39c12; font-size: 2rem; font-weight: bold; }
    .risk-gauge-high   { color: #e74c3c; font-size: 2rem; font-weight: bold; }
    .section-header {
        border-left: 4px solid #2d6a9f;
        padding-left: 12px;
        margin-bottom: 8px;
    }
    div[data-testid="stExpander"] { border-radius: 8px; }
</style>
""", unsafe_allow_html=True)

# ─────────────────────────────────────────────────────────────────────────────
# SESSION STATE DEFAULTS
# ─────────────────────────────────────────────────────────────────────────────
for key in ["model", "feature_cols", "scaler", "df_features", "label_map",
            "trained", "model_name", "threshold"]:
    if key not in st.session_state:
        st.session_state[key] = None
if "trained" not in st.session_state:
    st.session_state["trained"] = False


# ─────────────────────────────────────────────────────────────────────────────
# ── DATA LOADING ──────────────────────────────────────────────────────────────
# ─────────────────────────────────────────────────────────────────────────────
def load_weather_csv(file) -> pd.DataFrame:
    """
    Handles two common NOAA layouts:
      Layout A – header row 0 is the station meta-line (may contain commas),
                 actual column names are at row 1.
      Layout B – standard CSV with column names directly at row 0.
    Detects the layout by reading raw text lines — never lets pandas tokenize
    the ambiguous first row, which avoids the 'Expected N fields, saw M' error.
    """
    # Read raw bytes so we can seek back reliably regardless of file type
    raw_bytes = file.read()
    file.seek(0)

    # Decode to text and split into lines for layout detection
    text = raw_bytes.decode("utf-8", errors="replace")
    lines = text.splitlines()

    # Layout A heuristic: first line does NOT look like a CSV header.
    # A header row will contain a recognisable date/weather keyword.
    # A station meta-line looks like "BLACKSBURG ..., VA US (USC00440766)".
    HEADER_KEYWORDS = {"date", "tmax", "tmin", "tavg", "prcp", "snow", "snwd",
                       "datetime", "dt", "temp", "precip"}
    first_line_lower = lines[0].lower() if lines else ""
    first_line_has_header_kw = any(kw in first_line_lower for kw in HEADER_KEYWORDS)

    if not first_line_has_header_kw:
        # Layout A — skip the station meta-line; real header is on row 1
        df = pd.read_csv(io.StringIO(text), header=1, on_bad_lines="skip")
    else:
        # Layout B — standard header on row 0
        df = pd.read_csv(io.StringIO(text), header=0, on_bad_lines="skip")

    # Normalise column names
    df.columns = [c.strip() for c in df.columns]

    # Find the date column (case-insensitive)
    date_col = next((c for c in df.columns if c.lower() in ["date", "datetime", "dt"]), None)
    if date_col is None:
        raise ValueError("No date column found in weather CSV. "
                         "Expected a column named 'Date', 'datetime', or 'dt'.")
    df[date_col] = pd.to_datetime(df[date_col], errors="coerce")
    df = df.dropna(subset=[date_col])
    df = df.rename(columns={date_col: "date"})
    df = df.sort_values("date").reset_index(drop=True)

    # Shorten verbose column names
    rename_map = {}
    for c in df.columns:
        cu = c.upper()
        if "TAVG" in cu: rename_map[c] = "TAVG"
        elif "TMAX" in cu: rename_map[c] = "TMAX"
        elif "TMIN" in cu: rename_map[c] = "TMIN"
        elif "PRCP" in cu: rename_map[c] = "PRCP"
        elif "SNOW" in cu and "SNWD" not in cu: rename_map[c] = "SNOW"
        elif "SNWD" in cu: rename_map[c] = "SNWD"
    df = df.rename(columns=rename_map)

    # Coerce numeric columns
    weather_cols = [c for c in df.columns if c != "date"]
    for c in weather_cols:
        df[c] = pd.to_numeric(df[c], errors="coerce")

    return df


def load_storm_csv(file) -> pd.DataFrame:
    """
    Loads NOAA Storm Events export (or any CSV with EVENT_TYPE and BEGIN_DATE).
    Returns a DataFrame with a clean 'date' column.
    """
    df = pd.read_csv(file, low_memory=False)
    df.columns = [c.strip().upper() for c in df.columns]

    # Flexible date column detection
    date_col = next(
        (c for c in df.columns if c in ["BEGIN_DATE", "DATE", "BEGIN_DATETIME"]), None
    )
    if date_col is None:
        raise ValueError("No date column found in storm CSV. "
                         "Expected 'BEGIN_DATE', 'DATE', or 'BEGIN_DATETIME'.")

    df["date"] = pd.to_datetime(df[date_col], errors="coerce")
    df = df.dropna(subset=["date"])
    df["date"] = df["date"].dt.normalize()  # strip time → date only

    event_col = next((c for c in df.columns if "EVENT_TYPE" in c), None)
    if event_col:
        df = df.rename(columns={event_col: "EVENT_TYPE"})
    else:
        df["EVENT_TYPE"] = "Unknown"

    return df


# ─────────────────────────────────────────────────────────────────────────────
# ── FEATURE ENGINEERING ───────────────────────────────────────────────────────
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

def build_label(weather_df: pd.DataFrame, storm_df: pd.DataFrame,
                severe_types: set, min_category: str = "any") -> pd.DataFrame:
    """
    Creates a binary target: 1 = severe event on that day, 0 = none.
    min_category controls which event types count:
      'any'     – all events in the storm data
      'severe'  – only SEVERE_TYPES
    """
    if min_category == "severe":
        events = storm_df[storm_df["EVENT_TYPE"].str.lower().isin(severe_types)]
    else:
        events = storm_df

    event_dates = set(events["date"].dt.normalize())
    weather_df = weather_df.copy()
    weather_df["severe_event"] = weather_df["date"].isin(event_dates).astype(int)
    return weather_df


def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Full feature engineering pipeline:
      1. Calendar features
      2. Temperature-derived features
      3. Lag features (1, 3, 7 days)
      4. Rolling statistics (7-day, 14-day)
      5. Trend features
      6. Categorical encoding (month, season)
    """
    df = df.copy().sort_values("date").reset_index(drop=True)

    # ── 1. Calendar ──────────────────────────────────────────────────────────
    df["month"]      = df["date"].dt.month
    df["day_of_year"]= df["date"].dt.dayofyear
    df["week"]       = df["date"].dt.isocalendar().week.astype(int)
    df["season"] = df["month"].map({
        12:0, 1:0, 2:0,   # Winter
         3:1, 4:1, 5:1,   # Spring
         6:2, 7:2, 8:2,   # Summer
         9:3,10:3,11:3,   # Autumn
    })
    # Cyclical encoding of month and day-of-year
    df["month_sin"]   = np.sin(2 * np.pi * df["month"] / 12)
    df["month_cos"]   = np.cos(2 * np.pi * df["month"] / 12)
    df["doy_sin"]     = np.sin(2 * np.pi * df["day_of_year"] / 365)
    df["doy_cos"]     = np.cos(2 * np.pi * df["day_of_year"] / 365)

    # ── 2. Temperature-derived ────────────────────────────────────────────────
    base_cols = [c for c in ["TMAX","TMIN","TAVG","PRCP","SNOW","SNWD"] if c in df.columns]

    if "TMAX" in df.columns and "TMIN" in df.columns:
        df["temp_range"]   = df["TMAX"] - df["TMIN"]
        df["temp_mean_est"]= (df["TMAX"] + df["TMIN"]) / 2
    if "TAVG" in df.columns:
        # Approximate heat index (simplified Rothfusz)
        df["heat_index"] = df["TAVG"].apply(
            lambda t: t + 0.33 * 0.06 * (t - 14.5) if pd.notna(t) and t > 60 else t
        )
    if "PRCP" in df.columns:
        df["prcp_nonzero"] = (df["PRCP"] > 0).astype(int)
        df["heavy_rain"]   = (df["PRCP"] > 1.0).astype(int)
    if "SNOW" in df.columns:
        df["snow_nonzero"] = (df["SNOW"] > 0).astype(int)

    # ── 3. Lag features ───────────────────────────────────────────────────────
    for col in base_cols:
        for lag in [1, 2, 3, 7]:
            df[f"{col}_lag{lag}"] = df[col].shift(lag)

    # ── 4. Rolling statistics ─────────────────────────────────────────────────
    for col in base_cols:
        for window in [7, 14, 30]:
            df[f"{col}_roll{window}_mean"] = df[col].shift(1).rolling(window).mean()
            df[f"{col}_roll{window}_std"]  = df[col].shift(1).rolling(window).std()
            df[f"{col}_roll{window}_max"]  = df[col].shift(1).rolling(window).max()

    # ── 5. Trend features (difference from rolling mean) ─────────────────────
    for col in ["TMAX", "TMIN", "PRCP"] if all(c in df.columns for c in ["TMAX","TMIN","PRCP"]) else []:
        if f"{col}_roll7_mean" in df.columns:
            df[f"{col}_dev7"]  = df[col] - df[f"{col}_roll7_mean"]
            df[f"{col}_dev14"] = df[col] - df[f"{col}_roll14_mean"]

    # ── 6. Consecutive-day counters ───────────────────────────────────────────
    if "PRCP" in df.columns:
        # days since last rain
        df["days_since_rain"] = (
            df["PRCP"]
            .eq(0)
            .groupby((df["PRCP"] != 0).cumsum())
            .cumcount()
        )
    if "SNOW" in df.columns:
        df["days_since_snow"] = (
            df["SNOW"]
            .eq(0)
            .groupby((df["SNOW"] != 0).cumsum())
            .cumcount()
        )

    return df


# ─────────────────────────────────────────────────────────────────────────────
# ── MODEL TRAINING ────────────────────────────────────────────────────────────
# ─────────────────────────────────────────────────────────────────────────────
MODEL_REGISTRY = {
    "Random Forest": RandomForestClassifier(
        n_estimators=300, max_depth=12, min_samples_leaf=5,
        class_weight="balanced", random_state=42, n_jobs=-1,
    ),
    "Gradient Boosting": HistGradientBoostingClassifier(
        max_iter=300, max_depth=6, learning_rate=0.05,
        class_weight="balanced", random_state=42,
    ),
    "Logistic Regression": LogisticRegression(
        C=1.0, max_iter=1000, class_weight="balanced",
        solver="lbfgs", random_state=42,
    ),
}


def train_model(df: pd.DataFrame, model_name: str, test_size: float,
                threshold: float):
    target = "severe_event"
    drop_cols = {target, "date"} | {c for c in df.columns if "EVENT_TYPE" in c.upper()}
    feature_cols = [c for c in df.columns if c not in drop_cols]

    X = df[feature_cols].copy()
    y = df[target].copy()

    # Drop columns that are entirely NaN — SimpleImputer silently removes them
    # from the output array without adjusting the column list, which causes a
    # shape mismatch when rebuilding the DataFrame.
    all_nan_cols = [c for c in X.columns if X[c].isna().all()]
    if all_nan_cols:
        X = X.drop(columns=all_nan_cols)
    feature_cols = X.columns.tolist()

    # Impute remaining missing values — output shape now matches feature_cols
    imputer = SimpleImputer(strategy="median")
    X_imp = imputer.fit_transform(X)
    X_imp = pd.DataFrame(X_imp, columns=feature_cols)

    X_train, X_test, y_train, y_test = train_test_split(
        X_imp, y, test_size=test_size, shuffle=False  # temporal split
    )

    model = MODEL_REGISTRY[model_name]

    # Wrap Logistic Regression in scaler pipeline
    if model_name == "Logistic Regression":
        pipeline = Pipeline([
            ("scaler", StandardScaler()),
            ("clf", model),
        ])
        pipeline.fit(X_train, y_train)
        probs  = pipeline.predict_proba(X_test)[:, 1]
        fitted = pipeline
    else:
        model.fit(X_train, y_train)
        probs  = model.predict_proba(X_test)[:, 1]
        fitted = model

    preds = (probs >= threshold).astype(int)

    # Metrics
    cm      = confusion_matrix(y_test, preds)
    roc_auc = roc_auc_score(y_test, probs)
    ap      = average_precision_score(y_test, probs)
    report  = classification_report(y_test, preds, output_dict=True)

    # ROC & PR curves
    fpr, tpr, roc_thresh = roc_curve(y_test, probs)
    prec, rec, pr_thresh  = precision_recall_curve(y_test, probs)

    # Feature importance
    if model_name == "Random Forest":
        importances = model.feature_importances_
    elif model_name == "Gradient Boosting":
        importances = model.feature_importances_
    else:
        importances = np.abs(pipeline.named_steps["clf"].coef_[0])

    fi = pd.Series(importances, index=feature_cols).sort_values(ascending=False)

    return {
        "model":        fitted,
        "imputer":      imputer,
        "feature_cols": feature_cols,
        "X_test":       X_test,
        "y_test":       y_test,
        "probs":        probs,
        "preds":        preds,
        "cm":           cm,
        "roc_auc":      roc_auc,
        "ap":           ap,
        "report":       report,
        "fpr":          fpr,
        "tpr":          tpr,
        "prec":         prec,
        "rec":          rec,
        "fi":           fi,
        "threshold":    threshold,
    }


# ─────────────────────────────────────────────────────────────────────────────
# ── PREDICTION ────────────────────────────────────────────────────────────────
# ─────────────────────────────────────────────────────────────────────────────
def predict_single_day(result: dict, df_features: pd.DataFrame, new_row: dict) -> float:
    """Given raw weather values for ONE new day, engineer features and predict."""
    # Build a new row appended to historical data so lags are valid
    new_df = pd.DataFrame([new_row])
    new_df["date"] = pd.to_datetime(new_row["date"])
    new_df["severe_event"] = 0  # placeholder

    # Re-engineer on the combined history + new row
    combined = pd.concat([df_features[["date","TMAX","TMIN","TAVG","PRCP","SNOW","SNWD","severe_event"]
                                       if all(c in df_features.columns for c in ["TMAX","TMIN","TAVG","PRCP","SNOW","SNWD"])
                                       else df_features.columns.tolist()],
                          new_df], ignore_index=True)
    combined_eng = engineer_features(combined)
    last_row = combined_eng.iloc[[-1]][result["feature_cols"]]

    imputed = result["imputer"].transform(last_row)
    prob = result["model"].predict_proba(imputed)[0, 1]
    return prob


# ─────────────────────────────────────────────────────────────────────────────
# ── UI HELPERS ────────────────────────────────────────────────────────────────
# ─────────────────────────────────────────────────────────────────────────────
def risk_badge(prob: float) -> str:
    pct = prob * 100
    if prob < 0.30:
        return f'<span class="risk-gauge-low">🟢 LOW — {pct:.1f}%</span>'
    elif prob < 0.60:
        return f'<span class="risk-gauge-medium">🟡 MODERATE — {pct:.1f}%</span>'
    else:
        return f'<span class="risk-gauge-high">🔴 HIGH — {pct:.1f}%</span>'


def plot_confusion(cm):
    labels = ["No Event", "Severe Event"]
    fig = px.imshow(
        cm, text_auto=True,
        x=labels, y=labels,
        color_continuous_scale="Blues",
        labels=dict(x="Predicted", y="Actual"),
        title="Confusion Matrix",
    )
    fig.update_traces(textfont_size=18)
    fig.update_layout(height=340, margin=dict(t=50, b=0))
    return fig


def plot_roc(fpr, tpr, roc_auc):
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=fpr, y=tpr, mode="lines",
                             name=f"ROC AUC = {roc_auc:.3f}",
                             line=dict(color="#2d6a9f", width=2)))
    fig.add_trace(go.Scatter(x=[0,1], y=[0,1], mode="lines",
                             name="Random", line=dict(dash="dash", color="gray")))
    fig.update_layout(title="ROC Curve", xaxis_title="FPR",
                      yaxis_title="TPR", height=340,
                      margin=dict(t=50, b=0), legend=dict(x=0.6, y=0.1))
    return fig


def plot_pr(prec, rec, ap):
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=rec, y=prec, mode="lines",
                             name=f"AP = {ap:.3f}",
                             line=dict(color="#e74c3c", width=2)))
    fig.update_layout(title="Precision-Recall Curve", xaxis_title="Recall",
                      yaxis_title="Precision", height=340,
                      margin=dict(t=50, b=0), legend=dict(x=0.6, y=0.9))
    return fig


def plot_feature_importance(fi: pd.Series, top_n: int = 20):
    top = fi.head(top_n).sort_values()
    fig = px.bar(
        x=top.values, y=top.index,
        orientation="h",
        title=f"Top {top_n} Feature Importances",
        labels={"x": "Importance", "y": "Feature"},
        color=top.values,
        color_continuous_scale="Blues",
    )
    fig.update_layout(height=480, margin=dict(t=50, b=0), showlegend=False)
    return fig


def plot_risk_timeline(df: pd.DataFrame, probs_col: str = "risk_score", tail: int = 365):
    sub = df.tail(tail).copy()
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=sub["date"], y=sub[probs_col], mode="lines",
        name="Risk Score", line=dict(color="#2d6a9f"),
        fill="tozeroy", fillcolor="rgba(45,106,159,0.15)",
    ))
    if "severe_event" in sub.columns:
        events = sub[sub["severe_event"] == 1]
        fig.add_trace(go.Scatter(
            x=events["date"], y=events[probs_col],
            mode="markers", name="Actual Event",
            marker=dict(color="red", size=6, symbol="x"),
        ))
    fig.add_hline(y=0.30, line_dash="dot", line_color="#f39c12",
                  annotation_text="Moderate threshold")
    fig.add_hline(y=0.60, line_dash="dot", line_color="#e74c3c",
                  annotation_text="High threshold")
    fig.update_layout(
        title="Historical Risk Score Timeline",
        xaxis_title="Date", yaxis_title="Risk Score",
        height=400, margin=dict(t=50, b=0),
        yaxis=dict(range=[0, 1]),
    )
    return fig


def plot_monthly_risk(df: pd.DataFrame, probs_col: str = "risk_score"):
    if "month" not in df.columns:
        df["month"] = df["date"].dt.month
    monthly = df.groupby("month")[probs_col].mean().reset_index()
    month_names = {1:"Jan",2:"Feb",3:"Mar",4:"Apr",5:"May",6:"Jun",
                   7:"Jul",8:"Aug",9:"Sep",10:"Oct",11:"Nov",12:"Dec"}
    monthly["month_name"] = monthly["month"].map(month_names)
    fig = px.bar(
        monthly, x="month_name", y=probs_col,
        title="Average Risk Score by Month",
        color=probs_col, color_continuous_scale="RdYlGn_r",
        labels={"risk_score": "Avg Risk", "month_name": "Month"},
    )
    fig.update_layout(height=340, margin=dict(t=50, b=0))
    return fig


# ─────────────────────────────────────────────────────────────────────────────
# ── MAIN APP ──────────────────────────────────────────────────────────────────
# ─────────────────────────────────────────────────────────────────────────────
st.title("⛈️ Severe Weather Risk Predictor")
st.caption("Upload your weather & storm event datasets → engineer features → train an ML model → predict risk.")

# ── SIDEBAR ───────────────────────────────────────────────────────────────────
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
    min_category = "any" if label_mode.startswith("All") else "severe"

    st.divider()

    st.subheader("🤖 Model Settings")
    model_name = st.selectbox(
        "Algorithm", list(MODEL_REGISTRY.keys()), index=0
    )
    test_size  = st.slider("Test set size (%)", 10, 40, 20) / 100
    threshold  = st.slider(
        "Decision threshold", 0.1, 0.9, 0.40, 0.05,
        help="Probability above which a day is flagged as 'severe'."
    )

    run_btn = st.button("🚀 Train Model", type="primary", use_container_width=True)

    st.divider()
    st.caption("Built with scikit-learn • Streamlit • Plotly")

# ─────────────────────────────────────────────────────────────────────────────
# TAB LAYOUT
# ─────────────────────────────────────────────────────────────────────────────
tab_data, tab_features, tab_train, tab_eval, tab_predict = st.tabs([
    "📊 Data Explorer",
    "🔧 Feature Engineering",
    "🏋️ Training",
    "📈 Evaluation",
    "🔮 Predict New Day",
])

# ────────────────────────────── TAB: DATA EXPLORER ───────────────────────────
with tab_data:
    if not weather_file or not storm_file:
        st.info("👈 Upload both CSV files in the sidebar to get started.")
        st.markdown("""
        **Expected file formats:**

        | File | Required columns |
        |------|-----------------|
        | Daily Weather CSV | `Date`, `TMAX`, `TMIN`, `PRCP` (others optional: `TAVG`, `SNOW`, `SNWD`) |
        | Storm Events CSV | `BEGIN_DATE` (or `DATE`), `EVENT_TYPE` |

        Both NOAA GHCN weather exports and NOAA Storm Events Database exports are supported out-of-the-box.
        """)
    else:
        try:
            df_weather = load_weather_csv(weather_file)
            df_storm   = load_storm_csv(storm_file)
            st.session_state["df_weather"] = df_weather
            st.session_state["df_storm"]   = df_storm

            col1, col2 = st.columns(2)
            with col1:
                st.markdown("### 🌡️ Weather Data")
                st.metric("Rows", f"{len(df_weather):,}")
                st.metric("Date range", f"{df_weather['date'].min().date()} → {df_weather['date'].max().date()}")
                st.dataframe(df_weather.head(10), use_container_width=True)

                # Missing value heatmap
                miss = df_weather.isnull().mean().reset_index()
                miss.columns = ["column", "missing_pct"]
                miss = miss[miss["column"] != "date"]
                fig_miss = px.bar(miss, x="column", y="missing_pct",
                                  title="Missing Value Rate per Column",
                                  labels={"missing_pct":"Missing %","column":""},
                                  color="missing_pct", color_continuous_scale="Reds")
                fig_miss.update_layout(height=280, margin=dict(t=50,b=0))
                st.plotly_chart(fig_miss, use_container_width=True)

            with col2:
                st.markdown("### 🌪️ Storm Events Data")
                st.metric("Rows", f"{len(df_storm):,}")
                st.metric("Unique event types", df_storm["EVENT_TYPE"].nunique())
                st.dataframe(df_storm[["date","EVENT_TYPE"]].head(20), use_container_width=True)

                # Event type breakdown
                vc = df_storm["EVENT_TYPE"].value_counts().reset_index()
                vc.columns = ["EVENT_TYPE","count"]
                fig_ev = px.bar(vc.head(15), x="count", y="EVENT_TYPE",
                                orientation="h",
                                title="Top Event Types",
                                color="count", color_continuous_scale="Blues")
                fig_ev.update_layout(height=350, margin=dict(t=50,b=0), yaxis=dict(autorange="reversed"))
                st.plotly_chart(fig_ev, use_container_width=True)

            # Overlap visualisation
            st.markdown("### 📅 Event Frequency by Year")
            df_storm["year"] = df_storm["date"].dt.year
            by_year = df_storm.groupby(["year","EVENT_TYPE"]).size().reset_index(name="count")
            fig_yr = px.bar(by_year, x="year", y="count", color="EVENT_TYPE",
                            title="Storm Events by Year (stacked)")
            fig_yr.update_layout(height=360, margin=dict(t=50,b=0))
            st.plotly_chart(fig_yr, use_container_width=True)

        except Exception as e:
            st.error(f"Error loading files: {e}")

# ──────────────────────────── TAB: FEATURE ENGINEERING ───────────────────────
with tab_features:
    if "df_weather" not in st.session_state:
        st.info("Load your data first (Data Explorer tab).")
    else:
        df_weather = st.session_state["df_weather"]
        df_storm   = st.session_state["df_storm"]

        st.markdown("### 🏗️ Feature Engineering Pipeline")
        with st.expander("What features are created?", expanded=True):
            st.markdown("""
| Category | Features |
|----------|---------|
| **Calendar** | `month`, `season`, `day_of_year`, `week`, cyclical sin/cos encodings |
| **Temperature** | `temp_range`, `temp_mean_est`, `heat_index` |
| **Precipitation flags** | `prcp_nonzero`, `heavy_rain`, `snow_nonzero` |
| **Lag features** | 1, 2, 3, 7-day lags for all weather variables |
| **Rolling stats** | 7-, 14-, 30-day rolling mean, std, max (computed on lagged data to prevent leakage) |
| **Trend** | Deviation from 7- and 14-day rolling mean |
| **Consecutive counters** | `days_since_rain`, `days_since_snow` |
""")

        with st.spinner("Engineering features…"):
            df_labeled   = build_label(df_weather, df_storm, SEVERE_TYPES, min_category)
            df_engineered= engineer_features(df_labeled)
            df_engineered= df_engineered.dropna(subset=["date"])

        st.session_state["df_features"] = df_engineered

        col1, col2, col3 = st.columns(3)
        col1.metric("Total features", len(df_engineered.columns) - 2)
        col2.metric("Severe event days", int(df_engineered["severe_event"].sum()))
        col3.metric("Class balance", f"{df_engineered['severe_event'].mean()*100:.1f}% positive")

        # Feature correlation with target
        num_cols = df_engineered.select_dtypes(include=[np.number]).columns.tolist()
        num_cols = [c for c in num_cols if c != "severe_event"]
        corr = df_engineered[num_cols + ["severe_event"]].corr()["severe_event"].drop("severe_event")
        top_corr = corr.abs().sort_values(ascending=False).head(25)
        corr_df  = pd.DataFrame({"feature": top_corr.index,
                                  "correlation": corr[top_corr.index]})
        fig_corr = px.bar(corr_df, x="correlation", y="feature",
                          orientation="h",
                          title="Top 25 Feature Correlations with Target",
                          color="correlation",
                          color_continuous_scale="RdBu",
                          range_color=[-0.5, 0.5])
        fig_corr.update_layout(height=550, margin=dict(t=50,b=0),
                                yaxis=dict(autorange="reversed"))
        st.plotly_chart(fig_corr, use_container_width=True)

        st.dataframe(df_engineered.tail(10), use_container_width=True)

# ─────────────────────────────── TAB: TRAINING ───────────────────────────────
with tab_train:
    if "df_features" not in st.session_state or st.session_state["df_features"] is None:
        st.info("Complete Feature Engineering tab first.")
    else:
        st.markdown("### 🏋️ Model Training")

        if run_btn:
            df_feat = st.session_state["df_features"].copy()

            with st.spinner(f"Training {model_name}…"):
                result = train_model(df_feat, model_name, test_size, threshold)

            st.session_state.update({
                "result":     result,
                "trained":    True,
                "model_name": model_name,
                "threshold":  threshold,
            })

            # Compute risk scores on full dataset.
            # Must use the exact feature_cols the model was trained on (already
            # has all-NaN columns removed and matches the imputer's expected input).
            feat_cols = result["feature_cols"]
            X_all = result["imputer"].transform(
                df_feat[feat_cols].fillna(df_feat[feat_cols].median(numeric_only=True))
            )
            df_feat["risk_score"] = result["model"].predict_proba(X_all)[:, 1]
            st.session_state["df_features_scored"] = df_feat

            st.success(f"✅ {model_name} trained! ROC-AUC = **{result['roc_auc']:.3f}**")

        if st.session_state.get("trained"):
            result = st.session_state["result"]
            c1, c2, c3, c4 = st.columns(4)
            rep = result["report"]
            c1.metric("ROC-AUC",      f"{result['roc_auc']:.3f}")
            c2.metric("Avg Precision", f"{result['ap']:.3f}")
            c3.metric("F1 (severe)",   f"{rep.get('1', rep.get(1,{})).get('f1-score', 0):.3f}")
            c4.metric("Precision (severe)", f"{rep.get('1', rep.get(1,{})).get('precision', 0):.3f}")

            # Timeline
            if "df_features_scored" in st.session_state:
                st.plotly_chart(
                    plot_risk_timeline(st.session_state["df_features_scored"]),
                    use_container_width=True,
                )
                col1, col2 = st.columns(2)
                with col1:
                    st.plotly_chart(
                        plot_monthly_risk(st.session_state["df_features_scored"]),
                        use_container_width=True,
                    )
                with col2:
                    st.plotly_chart(
                        plot_feature_importance(result["fi"]),
                        use_container_width=True,
                    )
        else:
            st.info("Click **🚀 Train Model** in the sidebar to start.")

# ──────────────────────────────── TAB: EVALUATION ────────────────────────────
with tab_eval:
    if not st.session_state.get("trained"):
        st.info("Train a model first.")
    else:
        result = st.session_state["result"]
        st.markdown("### 📈 Model Evaluation")

        col1, col2 = st.columns(2)
        with col1:
            st.plotly_chart(plot_confusion(result["cm"]), use_container_width=True)
        with col2:
            st.plotly_chart(plot_roc(result["fpr"], result["tpr"], result["roc_auc"]),
                            use_container_width=True)

        col3, col4 = st.columns(2)
        with col3:
            st.plotly_chart(plot_pr(result["prec"], result["rec"], result["ap"]),
                            use_container_width=True)
        with col4:
            # Probability distribution
            df_probs = pd.DataFrame({
                "probability": result["probs"],
                "actual": result["y_test"].values
            })
            fig_dist = px.histogram(
                df_probs, x="probability", color="actual",
                barmode="overlay", nbins=40,
                title="Predicted Probability Distribution",
                labels={"actual":"Severe Event","probability":"P(Severe)"},
                color_discrete_map={0:"steelblue", 1:"crimson"},
            )
            fig_dist.add_vline(x=result["threshold"], line_dash="dash",
                               annotation_text="Threshold")
            fig_dist.update_layout(height=340, margin=dict(t=50,b=0))
            st.plotly_chart(fig_dist, use_container_width=True)

        st.markdown("#### Classification Report")
        rep_df = pd.DataFrame(result["report"]).T
        st.dataframe(rep_df.style.format("{:.3f}"), use_container_width=True)

        # Threshold sensitivity
        st.markdown("#### Threshold Sensitivity")
        thresholds = np.linspace(0.1, 0.9, 80)
        metrics_list = []
        for t in thresholds:
            p = (result["probs"] >= t).astype(int)
            cr = classification_report(result["y_test"], p, output_dict=True, zero_division=0)
            metrics_list.append({
                "threshold": t,
                "precision": cr.get("1", cr.get(1,{})).get("precision", 0),
                "recall":    cr.get("1", cr.get(1,{})).get("recall", 0),
                "f1":        cr.get("1", cr.get(1,{})).get("f1-score", 0),
            })
        thr_df = pd.DataFrame(metrics_list)
        fig_thr = go.Figure()
        for metric, color in [("precision","#2ecc71"),("recall","#e74c3c"),("f1","#3498db")]:
            fig_thr.add_trace(go.Scatter(x=thr_df["threshold"], y=thr_df[metric],
                                          mode="lines", name=metric.capitalize(),
                                          line=dict(color=color)))
        fig_thr.add_vline(x=result["threshold"], line_dash="dash",
                          annotation_text="Current threshold")
        fig_thr.update_layout(title="Precision / Recall / F1 vs Threshold",
                               xaxis_title="Threshold", yaxis_title="Score",
                               height=360, margin=dict(t=50,b=0))
        st.plotly_chart(fig_thr, use_container_width=True)

# ─────────────────────────────── TAB: PREDICT ────────────────────────────────
with tab_predict:
    if not st.session_state.get("trained"):
        st.info("Train a model first.")
    else:
        st.markdown("### 🔮 Predict Risk for a New Day")
        st.caption("Enter tomorrow's weather forecast to get a severe weather risk score.")

        df_feat = st.session_state.get("df_features_scored",
                  st.session_state.get("df_features"))

        # Determine available columns
        has = lambda c: c in df_feat.columns

        with st.form("predict_form"):
            cols = st.columns(3)
            input_date = cols[0].date_input("Date", value=pd.Timestamp.today())

            tmax = cols[1].number_input("TMAX (°F)", value=70.0, step=0.5) if has("TMAX") else 70.0
            tmin = cols[2].number_input("TMIN (°F)", value=50.0, step=0.5) if has("TMIN") else 50.0

            cols2 = st.columns(3)
            tavg = cols2[0].number_input("TAVG (°F)", value=(tmax+tmin)/2, step=0.5) if has("TAVG") else (tmax+tmin)/2
            prcp = cols2[1].number_input("PRCP (in)", value=0.0, min_value=0.0, step=0.01) if has("PRCP") else 0.0
            snow = cols2[2].number_input("SNOW (in)", value=0.0, min_value=0.0, step=0.1) if has("SNOW") else 0.0
            snwd = st.number_input("SNWD – Snow Depth (in)", value=0.0, min_value=0.0, step=0.1) if has("SNWD") else 0.0

            submitted = st.form_submit_button("⚡ Predict Risk", type="primary")

        if submitted:
            new_row = {
                "date": str(input_date),
                "severe_event": 0,
            }
            if has("TMAX"): new_row["TMAX"] = tmax
            if has("TMIN"): new_row["TMIN"] = tmin
            if has("TAVG"): new_row["TAVG"] = tavg
            if has("PRCP"): new_row["PRCP"] = prcp
            if has("SNOW"): new_row["SNOW"] = snow
            if has("SNWD"): new_row["SNWD"] = snwd

            try:
                result = st.session_state["result"]
                prob   = predict_single_day(result, df_feat, new_row)

                st.markdown("---")
                st.markdown(f"## Risk Assessment for **{input_date}**")
                st.markdown(risk_badge(prob), unsafe_allow_html=True)

                # Gauge chart
                fig_gauge = go.Figure(go.Indicator(
                    mode  = "gauge+number+delta",
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
                            "line": {"color": "black", "width": 4},
                            "thickness": 0.75,
                            "value": result["threshold"] * 100,
                        },
                    },
                    number={"suffix": "%", "font": {"size": 40}},
                ))
                fig_gauge.update_layout(height=300, margin=dict(t=30,b=0))
                st.plotly_chart(fig_gauge, use_container_width=True)

                # Contextual advice
                st.markdown("#### 📋 Interpretation")
                if prob < 0.30:
                    st.success("Low risk. Standard precautions apply.")
                elif prob < 0.60:
                    st.warning("Moderate risk. Monitor forecasts; consider activating early-warning protocols.")
                else:
                    st.error("High risk. Strong chance of severe weather. Recommend issuing alerts and activating emergency plans.")

                # Show comparable historical days
                st.markdown("#### 📚 Similar Historical Days")
                df_hist = df_feat.dropna(subset=["risk_score"])
                similar = df_hist.iloc[
                    (df_hist["risk_score"] - prob).abs().argsort()[:10]
                ][["date","risk_score","severe_event"] +
                  [c for c in ["TMAX","TMIN","PRCP","SNOW"] if c in df_hist.columns]]
                similar["risk_score"] = similar["risk_score"].round(3)
                st.dataframe(similar, use_container_width=True)

            except Exception as e:
                st.error(f"Prediction error: {e}")
