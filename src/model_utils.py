"""Shared helpers for ETA risk model training, evaluation, and inference."""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path

import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.metrics import classification_report, f1_score, precision_score, recall_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OrdinalEncoder
from xgboost import XGBClassifier

from features import CATEGORICAL_COLUMNS, FEATURE_COLUMNS


RANDOM_STATE = 42
MODEL_PATH = Path("models/eta_xgb_classifier_v1.pkl")
METRICS_PATH = Path("outputs/metrics.json")
SPLIT_DATE = pd.Timestamp("2015-02-10")
CLASS_LABELS = {0: "On-Time", 1: "Mod-Late", 2: "Breach"}
TARGET_THRESHOLDS = {
    "macro_f1": 0.55,
    "weighted_f1": 0.60,
    "class2_recall": 0.60,
    "class2_precision": 0.45,
}
RECOMMENDATIONS = {
    0: "No action required.",
    1: "Monitor — queue early customer notification.",
    2: "ALERT — reassign nearest available dasher immediately.",
}


def configure_console_output() -> None:
    """Switch stdout to UTF-8 so warning symbols print correctly on Windows."""

    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")


def ensure_output_directories() -> None:
    """Create the model and output folders if they do not exist."""

    os.makedirs("models", exist_ok=True)
    os.makedirs("outputs", exist_ok=True)


def build_preprocessor() -> ColumnTransformer:
    """Create the encoding pipeline used before XGBoost."""

    numeric_columns = [
        column_name
        for column_name in FEATURE_COLUMNS
        if column_name not in CATEGORICAL_COLUMNS
    ]
    preprocessor = ColumnTransformer(
        transformers=[
            (
                "cat",
                OrdinalEncoder(
                    handle_unknown="use_encoded_value",
                    unknown_value=-1,
                ),
                CATEGORICAL_COLUMNS,
            ),
            ("num", "passthrough", numeric_columns),
        ]
    )
    return preprocessor


def build_model() -> XGBClassifier:
    """Create the XGBoost classifier with the project settings."""

    model = XGBClassifier(
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
        random_state=RANDOM_STATE,
        n_jobs=-1,
        tree_method="hist",
    )
    return model


def build_pipeline() -> Pipeline:
    """Combine preprocessing and XGBoost into one reusable pipeline."""

    pipeline = Pipeline(
        steps=[
            ("preprocessor", build_preprocessor()),
            ("model", build_model()),
        ]
    )
    return pipeline


def compute_metrics(
    y_true,
    y_pred,
    cv_mean: float | None = None,
    cv_std: float | None = None,
) -> dict[str, float]:
    """Compute the core metrics requested by the project."""

    metrics = {
        "macro_f1": float(f1_score(y_true, y_pred, average="macro")),
        "weighted_f1": float(f1_score(y_true, y_pred, average="weighted")),
        "class2_recall": float(recall_score(y_true, y_pred, labels=[2], average=None)[0]),
        "class2_precision": float(
            precision_score(y_true, y_pred, labels=[2], average=None, zero_division=0)[0]
        ),
        "class2_f1": float(f1_score(y_true, y_pred, labels=[2], average=None)[0]),
        "cv_macro_f1_mean": None if cv_mean is None else float(cv_mean),
        "cv_macro_f1_std": None if cv_std is None else float(cv_std),
    }
    return metrics


def save_metrics(metrics: dict[str, float], metrics_path: Path = METRICS_PATH) -> None:
    """Persist metrics for reproducibility tracking across scripts."""

    ensure_output_directories()
    with open(metrics_path, "w", encoding="utf-8") as metrics_file:
        json.dump(metrics, metrics_file, indent=2)


def load_metrics(metrics_path: Path = METRICS_PATH) -> dict[str, float]:
    """Load saved metrics if they already exist."""

    if not metrics_path.exists():
        return {}
    with open(metrics_path, "r", encoding="utf-8") as metrics_file:
        return json.load(metrics_file)


def print_split_distribution(target_series, split_name: str) -> None:
    """Print class balance for one data split."""

    class_counts = target_series.value_counts().sort_index()
    class_percentages = target_series.value_counts(normalize=True).sort_index() * 100
    print(f"\n{split_name} class distribution")
    for class_id in class_counts.index:
        print(
            f"Class {class_id}: "
            f"count={int(class_counts[class_id]):,} "
            f"pct={class_percentages[class_id]:.2f}%"
        )


def print_classification_report(y_true, y_pred) -> None:
    """Print the labeled classification report for the test set."""

    report = classification_report(
        y_true,
        y_pred,
        target_names=[CLASS_LABELS[0], CLASS_LABELS[1], CLASS_LABELS[2]],
        zero_division=0,
    )
    print("\nClassification report")
    print(report)


def print_metric_table(metrics: dict[str, float]) -> None:
    """Print the metrics in the table format requested by the user."""

    rows = [
        ("Macro F1", metrics["macro_f1"]),
        ("Weighted F1", metrics["weighted_f1"]),
        ("Class 2 Recall", metrics["class2_recall"]),
        ("Class 2 Precision", metrics["class2_precision"]),
        ("Class 2 F1", metrics["class2_f1"]),
        ("CV Macro F1 (mean)", metrics["cv_macro_f1_mean"]),
        ("CV Macro F1 (std)", metrics["cv_macro_f1_std"]),
    ]

    print("\nMetric               | Value")
    print("---------------------|-------")
    for metric_name, metric_value in rows:
        if metric_value is None:
            print(f"{metric_name:<21}| n/a")
        else:
            print(f"{metric_name:<21}| {metric_value:.4f}")


def print_threshold_warnings(metrics: dict[str, float]) -> None:
    """Warn when the requested minimum performance targets are missed."""

    if metrics["macro_f1"] < TARGET_THRESHOLDS["macro_f1"]:
        print("⚠ WARNING Macro F1 < 0.55")
    if metrics["class2_recall"] < TARGET_THRESHOLDS["class2_recall"]:
        print("⚠ WARNING Class 2 Recall < 0.60")
    if metrics["class2_precision"] < TARGET_THRESHOLDS["class2_precision"]:
        print("⚠ WARNING Class 2 Precision < 0.45")
    if metrics["weighted_f1"] < TARGET_THRESHOLDS["weighted_f1"]:
        print("⚠ WARNING Weighted F1 < 0.60")
