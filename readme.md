# ETA Error Classification

Real-time multiclass ETA risk classification for food delivery orders using XGBoost.

## Objective

Predict delivery risk at the moment an order is placed so operations teams can intervene before an order goes late.

Classes:

| Class | Label | Rule |
|---|---|---|
| 0 | On-Time | `abs(actual_duration_mins - estimated_total_mins) <= 10` |
| 1 | Mod-Late | `eta_error_mins > 10 and eta_error_mins <= 15` |
| 2 | Breach | `eta_error_mins > 15` |

`eta_error_mins = actual_duration_mins - estimated_total_mins`

## Current Implementation

The project is implemented as a full Python pipeline in `src/`:

| File | Purpose |
|---|---|
| `src/features.py` | Loads `data/cleaned_data.csv`, engineers leakage-safe features, builds target labels, prints class distribution |
| `src/model_utils.py` | Shared constants, preprocessing pipeline, XGBoost configuration, metrics helpers |
| `src/train.py` | Chronological split, time-series cross-validation, model fitting, model save, metrics save |
| `src/evaluate.py` | Test-set metrics, confusion matrix, ROC curves, calibration curve, SHAP plots, final summary |
| `src/predict.py` | Scores a single raw order dictionary with the trained pipeline |

Outputs:

| Path | Artifact |
|---|---|
| `models/eta_xgb_classifier_v1.pkl` | Trained pipeline |
| `outputs/metrics.json` | Saved evaluation metrics |
| `outputs/confusion_matrix.png` | Confusion matrix heatmap |
| `outputs/roc_curves.png` | One-vs-rest ROC curves |
| `outputs/calibration_breach.png` | Class 2 calibration plot |
| `outputs/shap_bar_breach.png` | SHAP bar chart for breach class |
| `outputs/shap_beeswarm_breach.png` | SHAP beeswarm plot for breach class |

## Data

Input file: `data/cleaned_data.csv`

The code does not re-clean raw data. It loads the cleaned file directly and recomputes only the model features required for training and inference.

Important rule:

- `actual_delivery_time` is used only to derive the target label.
- `actual_delivery_time`, `actual_duration_mins`, and `eta_error_mins` are never used as model features.

## Engineered Features

All model features are available at order placement time:

- `hour_of_day`
- `day_of_week`
- `is_peak`
- `is_weekend`
- `dasher_utilization`
- `dasher_availability`
- `outstanding_per_dasher`
- `estimated_total_mins`
- `order_value_dollars`
- `price_range`
- `items_per_distinct`
- `load_critical`
- `market_id`
- `store_primary_category`
- `order_protocol`

Categorical columns are encoded with `OrdinalEncoder(handle_unknown='use_encoded_value', unknown_value=-1)`.

## Model

Model: `xgboost.XGBClassifier`

Configuration:

```python
XGBClassifier(
    n_estimators=500,
    learning_rate=0.05,
    max_depth=6,
    subsample=0.8,
    colsample_bytree=0.8,
    min_child_weight=50,
    objective="multi:softprob",
    num_class=3,
    eval_metric="mlogloss",
    use_label_encoder=False,
    random_state=42,
    n_jobs=-1,
    tree_method="hist",
)
```

Training details:

- Chronological split only
- Split date: `2015-02-10`
- Cross-validation: `TimeSeriesSplit(n_splits=5)`
- Imbalance handling: `compute_sample_weight(class_weight="balanced")`
- Random seed: `42`

Note: in this environment, `cross_val_score(..., n_jobs=-1)` raised a Windows process permission error, so the implementation uses `n_jobs=1` for cross-validation. Model training still uses XGBoost with `n_jobs=-1`.

## Environment Setup

Create and activate a virtual environment in PowerShell:

```powershell
cd "D:\projects\New folder\eta_delivery"
python -m venv .venv
.\.venv\Scripts\Activate.ps1
```

If activation is blocked:

```powershell
Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass
.\.venv\Scripts\Activate.ps1
```

Install dependencies:

```powershell
python -m pip install --upgrade pip
pip install -r requirements.txt
```

## Run Order

Run the scripts from the project root:

```powershell
python src\features.py
python src\train.py
python src\evaluate.py
python src\predict.py
```

## What Each Script Does

### `python src\features.py`

- Loads `data/cleaned_data.csv`
- Recomputes the target label
- Recomputes the project feature set
- Prints the overall class distribution

### `python src\train.py`

- Rebuilds features from the cleaned dataset
- Splits data using `created_at < 2015-02-10`
- Prints train and test class balance
- Runs time-series cross-validation
- Fits the full pipeline
- Evaluates on the held-out test set
- Saves model to `models/eta_xgb_classifier_v1.pkl`
- Saves metrics to `outputs/metrics.json`

### `python src\evaluate.py`

- Loads the saved pipeline
- Recreates the held-out test split
- Prints the classification report and metric table
- Saves confusion matrix, ROC curves, calibration curve, and SHAP plots
- Prints the final one-paragraph summary

### `python src\predict.py`

- Loads the saved model
- Builds features for one sample order
- Returns class, label, probabilities, and operations recommendation

## Sample Inference Output

The included sample order in `src/predict.py` scored as:

- Predicted class: `2`
- Label: `Breach`
- Breach probability: `0.8184`
- Recommendation: `ALERT — reassign nearest available dasher immediately.`

## Latest Observed Results

These are the metrics produced by the current implementation on this dataset:

| Metric | Value |
|---|---:|
| Macro F1 | 0.3831 |
| Weighted F1 | 0.7474 |
| Class 2 Recall | 0.6868 |
| Class 2 Precision | 0.9696 |
| Class 2 F1 | 0.8040 |
| CV Macro F1 Mean | 0.3206 |
| CV Macro F1 Std | 0.0017 |

Threshold check:

- `Macro F1 >= 0.55`: not met
- `Weighted F1 >= 0.60`: met
- `Class 2 Recall >= 0.60`: met
- `Class 2 Precision >= 0.45`: met

## Important Finding About This Dataset

The specified label rule produces a much more breach-heavy class distribution than the earlier project notes suggested.

Observed overall class distribution from the current pipeline:

| Class | Count | Percent |
|---|---:|---:|
| 0 | 5,653 | 2.88% |
| 1 | 13,931 | 7.10% |
| 2 | 176,744 | 90.02% |

This is the main reason the current model achieves strong Class 2 precision and recall but weak macro F1. The model is learning on a dataset where most orders are labeled as breach under the requested target definition.

## SHAP Findings

Top 5 features for Class 2 in the latest run:

1. `outstanding_per_dasher`
2. `hour_of_day`
3. `order_value_dollars`
4. `order_protocol`
5. `store_primary_category`

This aligns with the idea that queue pressure and timing are strong breach signals, but the exact ranking is from the trained model rather than assumed from EDA.

## Dependencies

See `requirements.txt`.

Current project requirements:

```text
pandas>=1.5
numpy>=1.23
scikit-learn>=1.2
xgboost>=1.7
shap>=0.41
matplotlib>=3.6
seaborn>=0.12
joblib>=1.2
```

## Notes

- The README previously referenced LightGBM and a preprocessing script; that is no longer accurate for this repository.
- The repository currently uses XGBoost end to end.
- All plots are saved to disk; no script uses `plt.show()`.
- The code is written so training and inference share the same feature logic.
