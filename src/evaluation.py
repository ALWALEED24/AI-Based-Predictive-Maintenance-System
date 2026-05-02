import os
import pandas as pd

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    roc_auc_score,
    average_precision_score
)


# ---------------------------------------------------------
# Helper: Safe metric calculation
# ---------------------------------------------------------
def calculate_basic_metrics(y_true, y_pred):
    """
    Calculate classification metrics for binary prediction.

    Positive class:
    1 = failure / anomaly
    0 = normal
    """

    cm = confusion_matrix(y_true, y_pred, labels=[0, 1])

    if cm.shape == (2, 2):
        true_negative, false_positive, false_negative, true_positive = cm.ravel()
    else:
        true_negative = false_positive = false_negative = true_positive = 0

    metrics = {
        "accuracy": accuracy_score(y_true, y_pred),
        "precision": precision_score(y_true, y_pred, zero_division=0),
        "recall": recall_score(y_true, y_pred, zero_division=0),
        "f1_score": f1_score(y_true, y_pred, zero_division=0),
        "true_negative": int(true_negative),
        "false_positive": int(false_positive),
        "false_negative": int(false_negative),
        "true_positive": int(true_positive),
    }

    return metrics


def calculate_auc_metrics(y_true, score_values):
    """
    Calculate ROC-AUC and PR-AUC safely.

    score_values must be higher for the positive class.
    Positive class:
    1 = failure / anomaly
    """

    auc_metrics = {
        "roc_auc": None,
        "pr_auc": None
    }

    try:
        if len(set(y_true)) > 1:
            auc_metrics["roc_auc"] = roc_auc_score(y_true, score_values)
            auc_metrics["pr_auc"] = average_precision_score(y_true, score_values)
    except Exception:
        auc_metrics["roc_auc"] = None
        auc_metrics["pr_auc"] = None

    return auc_metrics


def clean_metric_value(value):
    """
    Convert missing metric values to N/A-friendly output.
    """

    if value is None:
        return "N/A"

    try:
        if pd.isna(value):
            return "N/A"
    except Exception:
        pass

    return value


def build_metric_row(model_name, y_true, y_pred, score_values=None):
    """
    Build one row for the evaluation results table.
    """

    row = {
        "model": model_name
    }

    basic_metrics = calculate_basic_metrics(y_true, y_pred)
    row.update(basic_metrics)

    if score_values is not None:
        auc_metrics = calculate_auc_metrics(y_true, score_values)
        row.update(auc_metrics)
    else:
        row["roc_auc"] = None
        row["pr_auc"] = None

    return row


# ---------------------------------------------------------
# Random Forest metric handling
# ---------------------------------------------------------
def build_random_forest_row_from_metrics(rf_metrics):
    """
    Build Random Forest row from rf_metrics dictionary.

    This supports both old and improved versions of model_training.py.
    """

    row = {
        "model": "Random Forest",
        "accuracy": rf_metrics.get("accuracy", 0),
        "precision": rf_metrics.get("precision", 0),
        "recall": rf_metrics.get("recall", 0),
        "f1_score": rf_metrics.get("f1_score", 0),
        "roc_auc": rf_metrics.get("roc_auc", None),
        "pr_auc": rf_metrics.get("pr_auc", None),
        "true_negative": 0,
        "false_positive": 0,
        "false_negative": 0,
        "true_positive": 0,
    }

    confusion = rf_metrics.get("confusion_matrix", None)

    try:
        if confusion is not None:
            cm = pd.DataFrame(confusion).values

            if cm.shape == (2, 2):
                tn, fp, fn, tp = cm.ravel()
                row["true_negative"] = int(tn)
                row["false_positive"] = int(fp)
                row["false_negative"] = int(fn)
                row["true_positive"] = int(tp)
    except Exception:
        pass

    return row


def improve_random_forest_auc_from_saved_test_file(row):
    """
    If Random Forest ROC-AUC / PR-AUC are missing, try to calculate them
    from outputs/random_forest_test_results.csv.

    This file is created by main.py.
    """

    rf_test_path = os.path.join("outputs", "random_forest_test_results.csv")

    if not os.path.exists(rf_test_path):
        return row

    try:
        rf_test_df = pd.read_csv(rf_test_path)

        required_columns = ["actual_failure", "rf_prediction", "rf_failure_probability"]

        if all(col in rf_test_df.columns for col in required_columns):
            y_true = rf_test_df["actual_failure"]
            y_pred = rf_test_df["rf_prediction"]
            y_score = rf_test_df["rf_failure_probability"]

            updated_metrics = calculate_basic_metrics(y_true, y_pred)
            auc_metrics = calculate_auc_metrics(y_true, y_score)

            row.update(updated_metrics)

            if auc_metrics["roc_auc"] is not None:
                row["roc_auc"] = auc_metrics["roc_auc"]

            if auc_metrics["pr_auc"] is not None:
                row["pr_auc"] = auc_metrics["pr_auc"]

    except Exception:
        pass

    return row


# ---------------------------------------------------------
# Main Evaluation Function
# ---------------------------------------------------------
def evaluate_all_models(final_results, rf_metrics):
    """
    Evaluate all selected models.

    Models:
    1. Isolation Forest
       - Main anomaly detection model
       - Compared against Machine failure label as benchmark

    2. One-Class SVM
       - Unsupervised anomaly detection benchmark
       - Compared against Machine failure label as benchmark

    3. Random Forest
       - Supervised benchmark model
       - Uses labelled Machine failure target

    Output columns:
    model
    accuracy
    precision
    recall
    f1_score
    roc_auc
    pr_auc
    true_negative
    false_positive
    false_negative
    true_positive
    """

    if final_results is None or final_results.empty:
        return pd.DataFrame()

    if "Machine failure" not in final_results.columns:
        raise ValueError("Machine failure column is missing from final_results.")

    y_true = final_results["Machine failure"].astype(int)

    evaluation_rows = []

    # -----------------------------------------------------
    # Isolation Forest Evaluation
    # -----------------------------------------------------
    if "if_anomaly_label" in final_results.columns:
        y_pred_if = final_results["if_anomaly_label"].astype(int)

        # Isolation Forest decision_function:
        # higher score = more normal
        # lower score = more abnormal
        # For ROC-AUC / PR-AUC, positive class should have higher score.
        if "if_anomaly_score" in final_results.columns:
            if_score = -final_results["if_anomaly_score"]
        else:
            if_score = y_pred_if

        isolation_row = build_metric_row(
            model_name="Isolation Forest",
            y_true=y_true,
            y_pred=y_pred_if,
            score_values=if_score
        )

        evaluation_rows.append(isolation_row)

    # -----------------------------------------------------
    # One-Class SVM Evaluation
    # -----------------------------------------------------
    if "ocsvm_anomaly_label" in final_results.columns:
        y_pred_ocsvm = final_results["ocsvm_anomaly_label"].astype(int)

        # One-Class SVM decision_function:
        # higher score = more normal
        # lower score = more abnormal
        # For ROC-AUC / PR-AUC, positive class should have higher score.
        if "ocsvm_anomaly_score" in final_results.columns:
            ocsvm_score = -final_results["ocsvm_anomaly_score"]
        else:
            ocsvm_score = y_pred_ocsvm

        ocsvm_row = build_metric_row(
            model_name="One-Class SVM",
            y_true=y_true,
            y_pred=y_pred_ocsvm,
            score_values=ocsvm_score
        )

        evaluation_rows.append(ocsvm_row)

    # -----------------------------------------------------
    # Random Forest Evaluation
    # -----------------------------------------------------
    if rf_metrics is not None:
        rf_row = build_random_forest_row_from_metrics(rf_metrics)
        rf_row = improve_random_forest_auc_from_saved_test_file(rf_row)
        evaluation_rows.append(rf_row)

    evaluation_df = pd.DataFrame(evaluation_rows)

    # Keep column order stable for dashboard display
    column_order = [
        "model",
        "accuracy",
        "precision",
        "recall",
        "f1_score",
        "roc_auc",
        "pr_auc",
        "true_negative",
        "false_positive",
        "false_negative",
        "true_positive"
    ]

    for col in column_order:
        if col not in evaluation_df.columns:
            evaluation_df[col] = None

    evaluation_df = evaluation_df[column_order]

    # Clean missing values for CSV/dashboard readability
    for col in ["roc_auc", "pr_auc"]:
        evaluation_df[col] = evaluation_df[col].apply(clean_metric_value)

    return evaluation_df


# ---------------------------------------------------------
# Save Evaluation Results
# ---------------------------------------------------------
def save_evaluation_results(evaluation_df):
    """
    Save evaluation results to outputs/evaluation_results.csv.
    """

    os.makedirs("outputs", exist_ok=True)

    output_path = os.path.join("outputs", "evaluation_results.csv")
    evaluation_df.to_csv(output_path, index=False)

    print(f"Evaluation results saved to: {output_path}")