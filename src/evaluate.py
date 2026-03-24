"""Evaluate the trained ETA risk model and produce plots plus SHAP outputs."""

from __future__ import annotations

from pathlib import Path

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import shap
from sklearn.calibration import calibration_curve
from sklearn.metrics import confusion_matrix, roc_curve
from sklearn.preprocessing import label_binarize

from features import engineer_features, load_cleaned_data
from model_utils import (
    CLASS_LABELS,
    MODEL_PATH,
    RANDOM_STATE,
    SPLIT_DATE,
    compute_metrics,
    configure_console_output,
    ensure_output_directories,
    load_metrics,
    print_classification_report,
    print_metric_table,
    print_threshold_warnings,
    save_metrics,
)


def load_test_split() -> tuple[pd.DataFrame, pd.Series]:
    """Load the engineered test split using the training cutoff date."""

    raw_dataframe = load_cleaned_data()
    feature_dataframe, target_series = engineer_features(raw_dataframe)
    test_mask = raw_dataframe.loc[target_series.index, "created_at"] >= SPLIT_DATE
    x_test = feature_dataframe.loc[test_mask].copy()
    y_test = target_series.loc[test_mask].copy()
    return x_test, y_test


def format_confusion_annotations(confusion_counts: np.ndarray) -> np.ndarray:
    """Create count and row-percentage annotations for the heatmap."""

    row_percentages = confusion_counts / confusion_counts.sum(axis=1, keepdims=True).clip(min=1)
    annotation_matrix = np.empty_like(confusion_counts).astype(object)
    for row_index in range(confusion_counts.shape[0]):
        for column_index in range(confusion_counts.shape[1]):
            annotation_matrix[row_index, column_index] = (
                f"{confusion_counts[row_index, column_index]}\n"
                f"{row_percentages[row_index, column_index] * 100:.1f}%"
            )
    return annotation_matrix


def save_confusion_matrix(y_true: pd.Series, y_pred: np.ndarray) -> None:
    """Save the labeled confusion matrix heatmap."""

    confusion_counts = confusion_matrix(y_true, y_pred, labels=[0, 1, 2])
    annotation_matrix = format_confusion_annotations(confusion_counts)

    plt.figure(figsize=(8, 6))
    sns.heatmap(
        confusion_counts,
        annot=annotation_matrix,
        fmt="",
        cmap="Blues",
        xticklabels=[CLASS_LABELS[0], CLASS_LABELS[1], CLASS_LABELS[2]],
        yticklabels=[CLASS_LABELS[0], CLASS_LABELS[1], CLASS_LABELS[2]],
    )
    plt.title("ETA Risk Classification — Confusion Matrix")
    plt.xlabel("Predicted class")
    plt.ylabel("True class")
    plt.tight_layout()
    plt.savefig("outputs/confusion_matrix.png", dpi=200, bbox_inches="tight")
    plt.close()


def save_roc_curves(y_true: pd.Series, predicted_probabilities: np.ndarray) -> None:
    """Save one-vs-rest ROC curves for all three classes."""

    y_binary = label_binarize(y_true, classes=[0, 1, 2])

    plt.figure(figsize=(8, 6))
    for class_id in [0, 1, 2]:
        false_positive_rate, true_positive_rate, _ = roc_curve(
            y_binary[:, class_id],
            predicted_probabilities[:, class_id],
        )
        plt.plot(
            false_positive_rate,
            true_positive_rate,
            label=f"Class {class_id} ({CLASS_LABELS[class_id]})",
        )

    plt.plot([0, 1], [0, 1], linestyle="--", color="grey", linewidth=1)
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("ETA Risk Classification — ROC Curves")
    plt.legend()
    plt.tight_layout()
    plt.savefig("outputs/roc_curves.png", dpi=200, bbox_inches="tight")
    plt.close()


def save_calibration_curve(y_true: pd.Series, predicted_probabilities: np.ndarray) -> None:
    """Save the calibration curve for the breach class probability."""

    breach_truth = (y_true == 2).astype(int)
    breach_probability = predicted_probabilities[:, 2]
    fraction_of_positives, mean_predicted_value = calibration_curve(
        breach_truth,
        breach_probability,
        n_bins=10,
    )

    plt.figure(figsize=(8, 6))
    plt.plot(mean_predicted_value, fraction_of_positives, marker="o", label="Model")
    plt.plot([0, 1], [0, 1], linestyle="--", color="grey", label="Perfect calibration")
    plt.xlabel("Mean predicted breach probability")
    plt.ylabel("Observed breach rate")
    plt.title("Breach Probability Calibration")
    plt.legend()
    plt.tight_layout()
    plt.savefig("outputs/calibration_breach.png", dpi=200, bbox_inches="tight")
    plt.close()


def _normalize_shap_values(shap_values) -> np.ndarray:
    """Handle SHAP outputs across versions and return (n, f, c)."""

    if isinstance(shap_values, list):
        return np.stack(shap_values, axis=2)
    shap_array = np.asarray(shap_values)
    if shap_array.ndim == 3:
        return shap_array
    raise ValueError(f"Unexpected SHAP output shape: {shap_array.shape}")


def save_shap_outputs(pipeline, x_test: pd.DataFrame) -> dict[int, list[str]]:
    """Create SHAP plots and return the top features for each class."""

    transformed_test = pipeline.named_steps["preprocessor"].transform(x_test)
    feature_names = pipeline.named_steps["preprocessor"].get_feature_names_out()

    sample_size = min(2000, transformed_test.shape[0])
    rng = np.random.default_rng(RANDOM_STATE)
    sample_indices = rng.choice(transformed_test.shape[0], sample_size, replace=False)
    x_sample = transformed_test[sample_indices]

    xgb_model = pipeline.named_steps["model"]
    explainer = shap.TreeExplainer(xgb_model)
    shap_values = _normalize_shap_values(explainer.shap_values(x_sample))

    shap.summary_plot(
        shap_values[:, :, 2],
        x_sample,
        feature_names=feature_names,
        plot_type="bar",
        max_display=12,
        show=False,
    )
    plt.title("SHAP Feature Importance — Class 2 (Breach)")
    plt.savefig("outputs/shap_bar_breach.png", dpi=200, bbox_inches="tight")
    plt.close()

    shap.summary_plot(
        shap_values[:, :, 2],
        x_sample,
        feature_names=feature_names,
        max_display=12,
        show=False,
    )
    plt.title("SHAP Beeswarm — Class 2 (Breach)")
    plt.savefig("outputs/shap_beeswarm_breach.png", dpi=200, bbox_inches="tight")
    plt.close()

    top_features_by_class: dict[int, list[str]] = {}
    for class_id in [0, 1, 2]:
        mean_absolute_shap = np.abs(shap_values[:, :, class_id]).mean(axis=0)
        top_feature_indices = np.argsort(mean_absolute_shap)[::-1][:5]
        top_feature_names = [str(feature_names[index]) for index in top_feature_indices]
        top_features_by_class[class_id] = top_feature_names

        print(f"\nClass {class_id} top 5 features:")
        for feature_index in top_feature_indices:
            print(
                f"  {feature_names[feature_index]}: "
                f"{mean_absolute_shap[feature_index]:.4f}"
            )

    return top_features_by_class


def print_final_summary(metrics: dict[str, float], top_features_by_class: dict[int, list[str]]) -> None:
    """Print the requested one-paragraph summary of final model quality."""

    top_three_breach_features = ", ".join(top_features_by_class[2][:3])
    print(
        "\nFinal summary: "
        f"Macro F1 = {metrics['macro_f1']:.4f}, "
        f"Class 2 Recall = {metrics['class2_recall']:.4f}, "
        f"and the top 3 SHAP features for breach risk are {top_three_breach_features}."
    )


def main() -> None:
    """Load the trained model, evaluate it, and save all requested outputs."""

    np.random.seed(RANDOM_STATE)
    configure_console_output()
    ensure_output_directories()

    if not Path(MODEL_PATH).exists():
        raise FileNotFoundError(
            "Trained model not found. Run `python src/train.py` before evaluation."
        )

    pipeline = joblib.load(MODEL_PATH)
    x_test, y_test = load_test_split()

    y_pred = pipeline.predict(x_test)
    predicted_probabilities = pipeline.predict_proba(x_test)

    previous_metrics = load_metrics()
    metrics = compute_metrics(
        y_true=y_test,
        y_pred=y_pred,
        cv_mean=previous_metrics.get("cv_macro_f1_mean"),
        cv_std=previous_metrics.get("cv_macro_f1_std"),
    )

    print_classification_report(y_test, y_pred)
    print_metric_table(metrics)
    print_threshold_warnings(metrics)

    save_confusion_matrix(y_test, y_pred)
    save_roc_curves(y_test, predicted_probabilities)
    save_calibration_curve(y_test, predicted_probabilities)
    top_features_by_class = save_shap_outputs(pipeline, x_test)

    save_metrics(metrics)
    print("Metrics saved: outputs/metrics.json")
    print_final_summary(metrics, top_features_by_class)


if __name__ == "__main__":
    main()
