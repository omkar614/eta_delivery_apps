# AGENTS.md — ETA Error Classification Project

> This file tells Codex (and any AI coding agent) everything it needs to know about
> this project: what exists, what to build, how to run things, and what rules to follow.
> Keep this file updated as the project evolves.

---

## Project Summary

**Goal:** Classify food delivery orders into three ETA risk buckets at order placement time
so ops teams can intervene before a delivery goes late.

**Model:** XGBoost multiclass classifier (3 classes)
**Data:** DoorDash-style delivery records, Jan–Feb 2015, ~193K rows (already cleaned)
**Key metric:** Class 2 (Breach) Recall ≥ 0.60 — missing a breach is the costliest error

---

## Repository Layout

```
project-root/
│
├── data/
│   ├── cleaned_data.csv          ✅ READY — use this as model input
│   ├── historical_data.csv       📦 Raw archive — do not modify
│   └── kpi_summary.xlsx          📊 KPI reference from EDA phase
│
├── notebooks/
│   └── 01_cleaning.ipynb         ✅ Cleaning already done here
│
├── sql/                          📂 SQL queries (reference only for this task)
│
├── src/                          🔨 BUILD HERE — all model code goes in this folder
│   ├── features.py               → Feature engineering + target label creation
│   ├── train.py                  → Pipeline, XGBoost training, CV, model save
│   ├── evaluate.py               → Metrics, plots, SHAP explainability
│   └── predict.py                → Inference function for single orders
│
├── models/
│   └── eta_xgb_classifier_v1.pkl → Saved model (created by train.py)
│
├── outputs/                      → All plots and metrics saved here
│   ├── confusion_matrix.png
│   ├── roc_curves.png
│   ├── calibration_breach.png
│   ├── shap_bar_breach.png
│   ├── shap_beeswarm_breach.png
│   └── metrics.json
│
├── connect_db.py                 🔌 DB connection utility (existing — do not modify)
├── AGENTS.md                     📋 This file
└── README.md                     📖 Project documentation
```

---

## Current State

| Component | Status | Notes |
|-----------|--------|-------|
| Raw data | ✅ Done | historical_data.csv |
| Data cleaning | ✅ Done | cleaned_data.csv ready, 193,474 rows |
| EDA & KPIs | ✅ Done | 01_cleaning.ipynb, kpi_summary.xlsx |
| Feature engineering | 🔨 To build | src/features.py |
| Model training | 🔨 To build | src/train.py |
| Evaluation | 🔨 To build | src/evaluate.py |
| Inference | 🔨 To build | src/predict.py |

---

## Target Labels

| Class | Name | Rule |
|-------|------|------|
| 0 | On-Time | `abs(eta_error_mins) <= 10` |
| 1 | Moderately Late | `eta_error_mins > 10 and <= 15` |
| 2 | Breach | `eta_error_mins > 15` ← **priority class** |

`eta_error_mins = actual_duration_mins - estimated_total_mins`

---

## Features to Engineer

All features must be derived from data available **at order placement time only**.
`actual_delivery_time` is used solely to create the target label — never as a feature.

| Feature | Source |
|---------|--------|
| `hour_of_day` | `created_at.hour` |
| `day_of_week` | `created_at.dayofweek` |
| `is_peak` | hours 11–13 or 18–21 |
| `is_weekend` | dayofweek in [5, 6] |
| `dasher_utilization` | `busy / onshift`, clipped 0–2 |
| `dasher_availability` | `onshift - busy` |
| `outstanding_per_dasher` | `outstanding / onshift`, clipped 0–20 |
| `estimated_total_mins` | `(place_dur + drive_dur) / 60` |
| `order_value_dollars` | `subtotal / 100` |
| `price_range` | `max_item_price - min_item_price` |
| `items_per_distinct` | `total_items / max(1, num_distinct_items)` |
| `load_critical` | `outstanding_orders > 40` → 1/0 |
| `market_id` | categorical (encode) |
| `store_primary_category` | categorical (encode) |
| `order_protocol` | categorical (encode) |

---

## Model Spec

```python
from xgboost import XGBClassifier

XGBClassifier(
    n_estimators=500,
    learning_rate=0.05,
    max_depth=6,
    subsample=0.8,
    colsample_bytree=0.8,
    min_child_weight=50,
    objective='multi:softprob',
    num_class=3,
    eval_metric='mlogloss',
    use_label_encoder=False,
    random_state=42,
    n_jobs=-1,
    tree_method='hist'
)
```

Use `compute_sample_weight(class_weight='balanced')` on training labels.
Wrap in `sklearn.pipeline.Pipeline` with `OrdinalEncoder` for categorical columns.

---

## Train / Test Split

**Chronological split — never random shuffle.**

```python
split_date = pd.Timestamp('2015-02-10')
train = df[df['created_at'] < split_date]   # ~116K rows
test  = df[df['created_at'] >= split_date]  # ~77K rows
```

Cross-validate with `TimeSeriesSplit(n_splits=5)` on training data only.

---

## Performance Targets

| Metric | Minimum Target |
|--------|---------------|
| Macro F1 | ≥ 0.55 |
| Class 2 Recall | ≥ 0.60 ← most important |
| Class 2 Precision | ≥ 0.45 |
| Weighted F1 | ≥ 0.60 |

Print a `⚠ WARNING` line for any target that is not met.

---

## Script Run Order

```bash
python src/features.py      # verify feature engineering, print class distribution
python src/train.py         # train model, run CV, save to models/
python src/evaluate.py      # generate all plots and metrics.json
python src/predict.py       # test inference on sample order
```

---

## Coding Rules for Agent

0. **Write for an intern** — every function needs a short docstring explaining what it does and why; every non-obvious line needs an inline comment; avoid clever one-liners; prefer 3 readable lines over 1 cryptic one; variable names must be descriptive (`dasher_utilization` not `du`, `X_train` not `xt`)
1. **No data leakage** — `actual_delivery_time` is forbidden as a model feature
2. **Chronological split only** — never use `train_test_split(shuffle=True)` on this data
3. **random_state=42** everywhere — all random operations must be seeded
4. **Divide-by-zero** — handle with `.clip()`, not `try/except`
5. **No plt.show()** — all plots saved to `outputs/`, project may run headless
6. **Modular code** — each script has functions + a `if __name__ == '__main__':` block
7. **Do not touch** `data/historical_data.csv` or `connect_db.py`
8. **Input file** — always load from `data/cleaned_data.csv`, never re-clean
9. **outputs/ and models/** — create these directories if they don't exist (`os.makedirs(..., exist_ok=True)`)
10. **metrics.json** — always write evaluation results here for reproducibility tracking

---

## Key EDA Findings (context for agent)

- Actual avg delivery: **47.1 mins** | Platform ETA avg: **14.2 mins** — prep time is missing
- Worst delivery hour: **2 PM (59.6 mins avg)** — shoulder period, low dasher coverage
- Breach threshold: orders with **>40 outstanding** spike to 48.6 min avg delivery
- Worst market: **Market 1** (MAE 35.8) | Best: **Market 5** (MAE 31.3)
- Slowest food categories: **Pizza (50.0 min)**, **Japanese (50.8 min)**
- Current breach rate (baseline): **15.9%** — model should help reduce this via early intervention

---

## Dependencies

```
python>=3.9
pandas>=1.5
numpy>=1.23
scikit-learn>=1.2
xgboost>=1.7
shap>=0.41
matplotlib>=3.6
seaborn>=0.12
joblib>=1.2
```

Install: `pip install -r requirements.txt`