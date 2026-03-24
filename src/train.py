"""Train the ETA multiclass XGBoost model with a chronological split."""

from __future__ import annotations

import os

import joblib
import numpy as np
import pandas as pd
from sklearn.model_selection import TimeSeriesSplit, cross_val_score
from sklearn.utils.class_weight import compute_sample_weight

from features import engineer_features, load_cleaned_data
from model_utils import (
    MODEL_PATH,
    RANDOM_STATE,
    SPLIT_DATE,
    build_pipeline,
    compute_metrics,
    configure_console_output,
    ensure_output_directories,
    print_classification_report,
    print_metric_table,
    print_split_distribution,
    print_threshold_warnings,
    save_metrics,
)


def load_training_data() -> tuple[pd.DataFrame, pd.DataFrame, pd.Series]:
    """Load the cleaned dataset and engineer the modeling dataset."""

    raw_dataframe = load_cleaned_data()
    feature_dataframe, target_series = engineer_features(raw_dataframe)
    return raw_dataframe, feature_dataframe, target_series


def split_train_test(
    raw_dataframe: pd.DataFrame,
    feature_dataframe: pd.DataFrame,
    target_series: pd.Series,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    """Split the dataset chronologically to avoid leakage across time."""

    train_mask = raw_dataframe.loc[target_series.index, "created_at"] < SPLIT_DATE
    x_train = feature_dataframe.loc[train_mask].copy()
    x_test = feature_dataframe.loc[~train_mask].copy()
    y_train = target_series.loc[train_mask].copy()
    y_test = target_series.loc[~train_mask].copy()

    print(f"Split date: {SPLIT_DATE.date()}")
    print(f"Train rows: {len(x_train):,}")
    print(f"Test rows: {len(x_test):,}")
    print_split_distribution(y_train, "Train")
    print_split_distribution(y_test, "Test")
    return x_train, x_test, y_train, y_test


def cross_validate_pipeline(pipeline, x_train: pd.DataFrame, y_train: pd.Series) -> np.ndarray:
    """Run time-aware cross-validation on the training split only."""

    time_series_cv = TimeSeriesSplit(n_splits=5)
    cv_scores = cross_val_score(
        pipeline,
        x_train,
        y_train,
        cv=time_series_cv,
        scoring="f1_macro",
        n_jobs=1,
    )
    print(f"\nCV Macro F1: {cv_scores.mean():.4f} ± {cv_scores.std():.4f}")
    return cv_scores


def fit_pipeline(pipeline, x_train: pd.DataFrame, y_train: pd.Series):
    """Fit the pipeline on the full training set with balanced sample weights."""

    sample_weights = compute_sample_weight(class_weight="balanced", y=y_train)
    pipeline.fit(x_train, y_train, model__sample_weight=sample_weights)
    return pipeline


def save_pipeline(pipeline) -> None:
    """Save the trained model artifact to disk."""

    ensure_output_directories()
    joblib.dump(pipeline, MODEL_PATH)
    model_size_mb = os.path.getsize(MODEL_PATH) / 1e6
    print(f"\nModel saved: {MODEL_PATH} ({model_size_mb:.2f} MB)")


def main() -> None:
    """Run the full training workflow and persist the fitted model."""

    np.random.seed(RANDOM_STATE)
    configure_console_output()
    ensure_output_directories()

    raw_dataframe, feature_dataframe, target_series = load_training_data()
    x_train, x_test, y_train, y_test = split_train_test(
        raw_dataframe,
        feature_dataframe,
        target_series,
    )

    pipeline = build_pipeline()
    cv_scores = cross_validate_pipeline(pipeline, x_train, y_train)
    trained_pipeline = fit_pipeline(pipeline, x_train, y_train)

    y_test_pred = trained_pipeline.predict(x_test)
    metrics = compute_metrics(
        y_true=y_test,
        y_pred=y_test_pred,
        cv_mean=float(cv_scores.mean()),
        cv_std=float(cv_scores.std()),
    )

    print_classification_report(y_test, y_test_pred)
    print_metric_table(metrics)
    print_threshold_warnings(metrics)

    save_pipeline(trained_pipeline)
    save_metrics(metrics)
    print("Metrics saved: outputs/metrics.json")


if __name__ == "__main__":
    main()
