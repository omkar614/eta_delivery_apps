"""Real-time inference for a single ETA risk order."""

from __future__ import annotations

from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from features import engineer_features
from model_utils import (
    CLASS_LABELS,
    MODEL_PATH,
    RANDOM_STATE,
    RECOMMENDATIONS,
    configure_console_output,
)


def predict_order(pipeline, order_dict: dict) -> dict:
    """Score one raw order dictionary with the trained pipeline."""

    raw_order_dataframe = pd.DataFrame([order_dict])
    feature_dataframe = engineer_features(raw_order_dataframe, include_target=False)

    predicted_class = int(pipeline.predict(feature_dataframe)[0])
    class_probabilities = pipeline.predict_proba(feature_dataframe)[0]

    probability_map = {
        CLASS_LABELS[class_id]: float(class_probabilities[class_id])
        for class_id in range(len(class_probabilities))
    }
    prediction = {
        "class": predicted_class,
        "label": CLASS_LABELS[predicted_class],
        "probabilities": probability_map,
        "recommendation": RECOMMENDATIONS[predicted_class],
    }
    return prediction


def build_sample_order() -> dict:
    """Return the sample order requested in the project instructions."""

    return {
        "market_id": 1,
        "created_at": "2015-02-12 14:00:00",
        "store_primary_category": "pizza",
        "order_protocol": 1,
        "total_items": 3,
        "subtotal": 2800,
        "num_distinct_items": 2,
        "min_item_price": 500,
        "max_item_price": 1200,
        "total_onshift_dashers": 10,
        "total_busy_dashers": 9,
        "total_outstanding_orders": 45,
        "estimated_order_place_duration": 300,
        "estimated_store_to_consumer_driving_duration": 720,
    }


def main() -> None:
    """Load the model and score the provided sample order."""

    np.random.seed(RANDOM_STATE)
    configure_console_output()

    if not Path(MODEL_PATH).exists():
        raise FileNotFoundError(
            "Trained model not found. Run `python src/train.py` before inference."
        )

    pipeline = joblib.load(MODEL_PATH)
    sample_order = build_sample_order()
    prediction = predict_order(pipeline, sample_order)

    print("Prediction result")
    print(prediction)
    print(f"Ops recommendation: {prediction['recommendation']}")


if __name__ == "__main__":
    main()
