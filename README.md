# ⛈️ Severe Weather Risk Predictor

A plug-and-play Streamlit app that trains an ML classifier on your historical
weather & storm data and predicts the daily probability of a severe weather event.

## Quick Start

```bash
pip install -r requirements.txt
streamlit run app.py
```

## Input File Formats

### Daily Weather CSV
NOAA GHCN export format (or compatible):

| Column | Description |
|--------|-------------|
| `Date` | YYYY-MM-DD |
| `TMAX` | Max temperature (°F) |
| `TMIN` | Min temperature (°F) |
| `TAVG` | Avg temperature (°F) — optional |
| `PRCP` | Precipitation (inches) |
| `SNOW` | Snowfall (inches) — optional |
| `SNWD` | Snow depth (inches) — optional |

Column names can contain verbose suffixes like `TMAX (Degrees Fahrenheit)` — the
app strips those automatically.  The station meta-line header used by NOAA GHCN
exports is also handled automatically.

### Storm Events CSV
NOAA Storm Events Database export format (or compatible):

| Column | Description |
|--------|-------------|
| `BEGIN_DATE` | Event start date (MM/DD/YYYY or YYYY-MM-DD) |
| `EVENT_TYPE` | e.g. `Thunderstorm Wind`, `Flash Flood`, `Tornado` |

All other columns are optional (used for display only).

## Feature Engineering

The pipeline automatically creates 80+ features:

- **Calendar** — month, season, day-of-year, cyclical sin/cos encodings
- **Temperature** — temp range, estimated mean, simplified heat index
- **Precipitation flags** — binary rain/snow indicators, heavy-rain flag
- **Lag features** — 1, 2, 3, 7-day lags for all weather variables
- **Rolling statistics** — 7-, 14-, 30-day rolling mean / std / max
  (always computed on *shifted* data to prevent target leakage)
- **Trend features** — deviation from 7- and 14-day rolling mean
- **Consecutive counters** — days since last rain / snow

## Models

| Model | Notes |
|-------|-------|
| Random Forest | Default; handles non-linearity well, robust to outliers |
| Gradient Boosting | Often highest accuracy; slower to train |
| Logistic Regression | Fastest; good baseline; features are scaled automatically |

All models use `class_weight="balanced"` to handle the natural class imbalance
(severe weather days are rare).

## Workflow

1. **Data Explorer** — preview both datasets, check missing values, event distributions
2. **Feature Engineering** — see what features are created and their correlation with the target
3. **Training** — train the chosen model, view risk score timeline
4. **Evaluation** — ROC, PR curve, confusion matrix, threshold sensitivity
5. **Predict New Day** — enter forecast values and get a risk score + gauge
