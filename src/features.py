"""Feature engineering for ETA risk classification.

This module creates the leakage-safe model inputs that are available at
order placement time. It also builds the multiclass target label when
ground-truth delivery timestamps are present.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd


DATA_PATH = Path("data/cleaned_data.csv")
CATEGORICAL_COLUMNS = ["market_id", "store_primary_category", "order_protocol"]
FEATURE_COLUMNS = [
    "hour_of_day",
    "day_of_week",
    "is_peak",
    "is_weekend",
    "dasher_utilization",
    "dasher_availability",
    "outstanding_per_dasher",
    "estimated_total_mins",
    "order_value_dollars",
    "price_range",
    "items_per_distinct",
    "load_critical",
    "market_id",
    "store_primary_category",
    "order_protocol",
]


def load_cleaned_data(data_path: Path | str = DATA_PATH) -> pd.DataFrame:
    """Load the cleaned dataset and parse datetime columns consistently."""

    dataframe = pd.read_csv(
        data_path,
        parse_dates=["created_at", "actual_delivery_time"],
    )
    return dataframe


def assign_label(error_minutes: float) -> int:
    """Map ETA error to the project risk buckets."""

    if abs(error_minutes) <= 10:
        return 0
    if error_minutes <= 15:
        return 1
    return 2


def _safe_ratio(numerator: pd.Series, denominator: pd.Series) -> pd.Series:
    """Create a ratio while preventing divide-by-zero with clipping."""

    safe_denominator = denominator.clip(lower=1e-3)
    return numerator / safe_denominator


def engineer_features(
    dataframe: pd.DataFrame,
    include_target: bool = True,
) -> tuple[pd.DataFrame, pd.Series] | pd.DataFrame:
    """Build model features and, when available, the target label.

    The same function is used by training and real-time inference so the
    feature logic stays identical across the project.
    """

    working_dataframe = dataframe.copy()
    working_dataframe["created_at"] = pd.to_datetime(working_dataframe["created_at"])

    # Convert categorical fields to stable string tokens for encoding.
    working_dataframe["market_id"] = (
        working_dataframe["market_id"].fillna(-1).astype(int).astype(str)
    )
    working_dataframe["store_primary_category"] = (
        working_dataframe["store_primary_category"].fillna("Unknown").astype(str)
    )
    working_dataframe["order_protocol"] = (
        working_dataframe["order_protocol"].fillna(-1).astype(int).astype(str)
    )

    working_dataframe["hour_of_day"] = working_dataframe["created_at"].dt.hour
    working_dataframe["day_of_week"] = working_dataframe["created_at"].dt.dayofweek
    working_dataframe["is_peak"] = (
        working_dataframe["hour_of_day"].isin([11, 12, 13, 18, 19, 20, 21]).astype(int)
    )
    working_dataframe["is_weekend"] = (
        working_dataframe["day_of_week"].isin([5, 6]).astype(int)
    )

    working_dataframe["dasher_utilization"] = _safe_ratio(
        working_dataframe["total_busy_dashers"],
        working_dataframe["total_onshift_dashers"],
    ).clip(0, 2)
    working_dataframe["dasher_availability"] = (
        working_dataframe["total_onshift_dashers"]
        - working_dataframe["total_busy_dashers"]
    )
    working_dataframe["outstanding_per_dasher"] = _safe_ratio(
        working_dataframe["total_outstanding_orders"],
        working_dataframe["total_onshift_dashers"],
    ).clip(0, 20)
    working_dataframe["estimated_total_mins"] = (
        working_dataframe["estimated_order_place_duration"]
        + working_dataframe["estimated_store_to_consumer_driving_duration"]
    ) / 60.0
    working_dataframe["order_value_dollars"] = working_dataframe["subtotal"] / 100.0
    working_dataframe["price_range"] = (
        working_dataframe["max_item_price"] - working_dataframe["min_item_price"]
    )
    working_dataframe["items_per_distinct"] = working_dataframe["total_items"] / (
        working_dataframe["num_distinct_items"].clip(lower=1)
    )
    working_dataframe["load_critical"] = (
        working_dataframe["total_outstanding_orders"] > 40
    ).astype(int)

    feature_dataframe = working_dataframe.loc[:, FEATURE_COLUMNS].copy()

    if not include_target:
        return feature_dataframe

    working_dataframe["actual_delivery_time"] = pd.to_datetime(
        working_dataframe["actual_delivery_time"]
    )
    working_dataframe["actual_duration_mins"] = (
        working_dataframe["actual_delivery_time"] - working_dataframe["created_at"]
    ).dt.total_seconds() / 60.0
    working_dataframe["eta_error_mins"] = (
        working_dataframe["actual_duration_mins"]
        - working_dataframe["estimated_total_mins"]
    )

    # Remove any rows that cannot create a valid label before training.
    valid_rows = working_dataframe["eta_error_mins"].notna()
    feature_dataframe = feature_dataframe.loc[valid_rows].copy()
    target_series = working_dataframe.loc[valid_rows, "eta_error_mins"].apply(assign_label)
    target_series.name = "target"
    return feature_dataframe, target_series


def print_class_distribution(target_series: pd.Series, title: str) -> None:
    """Print counts and percentages for each class in a readable table."""

    class_counts = target_series.value_counts().sort_index()
    class_percentages = target_series.value_counts(normalize=True).sort_index() * 100

    print(f"\n{title}")
    for class_id in class_counts.index:
        print(
            f"Class {class_id}: "
            f"count={int(class_counts[class_id]):,} "
            f"pct={class_percentages[class_id]:.2f}%"
        )


def main() -> None:
    """Run feature engineering as a standalone verification script."""

    dataframe = load_cleaned_data()
    feature_dataframe, target_series = engineer_features(dataframe)

    print(f"Loaded rows: {len(dataframe):,}")
    print(f"Feature matrix shape: {feature_dataframe.shape}")
    print_class_distribution(target_series, "Overall class distribution")
    print("\nFeature preview:")
    print(feature_dataframe.head().to_string())


if __name__ == "__main__":
    np.random.seed(42)
    main()
