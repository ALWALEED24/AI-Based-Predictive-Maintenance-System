import pandas as pd
from sklearn.ensemble import IsolationForest, RandomForestClassifier
from sklearn.svm import OneClassSVM
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    roc_auc_score,
    average_precision_score,
)


# ---------------------------------------------------------
# Columns and Project Settings
# ---------------------------------------------------------
RAW_FEATURE_COLUMNS = [
    "Type",
    "Air temperature [K]",
    "Process temperature [K]",
    "Rotational speed [rpm]",
    "Torque [Nm]",
    "Tool wear [min]",
]

NUMERIC_SENSOR_COLUMNS = [
    "Air temperature [K]",
    "Process temperature [K]",
    "Rotational speed [rpm]",
    "Torque [Nm]",
    "Tool wear [min]",
]

TARGET_COLUMN = "Machine failure"
RANDOM_STATE = 42
TEST_SIZE = 0.20
ANOMALY_RATE = 0.03
SCALED_PREFIX = "scaled_"


# ---------------------------------------------------------
# Feature Helpers
# ---------------------------------------------------------
def get_model_feature_columns(data):
    """
    Select the correct feature columns for model training.

    Important update:
    - If scaled_* columns exist, the models use them.
    - Original sensor columns remain available for dashboard/report display.
    - If scaled columns do not exist yet, the function safely falls back to
      the original feature names to keep the old pipeline running.
    """

    feature_columns = []

    # Handle machine Type safely.
    # Priority:
    # 1. scaled_Type
    # 2. Type if numeric/encoded
    # 3. one-hot columns such as Type_H, Type_L, Type_M
    scaled_type_col = f"{SCALED_PREFIX}Type"

    if scaled_type_col in data.columns:
        feature_columns.append(scaled_type_col)
    elif "Type" in data.columns:
        feature_columns.append("Type")
    else:
        type_dummy_cols = sorted(
            [col for col in data.columns if col.startswith("Type_")]
        )
        feature_columns.extend(type_dummy_cols)

    # Handle numeric sensor features.
    # Prefer scaled columns for ML; keep original columns for human-readable output.
    for col in NUMERIC_SENSOR_COLUMNS:
        scaled_col = f"{SCALED_PREFIX}{col}"

        if scaled_col in data.columns:
            feature_columns.append(scaled_col)
        elif col in data.columns:
            feature_columns.append(col)

    return feature_columns


def get_display_feature_name(feature_name):
    """
    Convert model feature column names into readable names for reports/charts.
    Example:
    scaled_Torque [Nm] -> Torque [Nm]
    """

    if feature_name.startswith(SCALED_PREFIX):
        return feature_name.replace(SCALED_PREFIX, "", 1)

    return feature_name


def normalize_scores(values):
    """
    Min-max normalize a score series to a 0-1 range.

    This is used only for combining model risk indicators.
    It does not replace the original model scores.
    """

    series = pd.Series(values).astype(float)
    min_value = series.min()
    max_value = series.max()

    if pd.isna(min_value) or pd.isna(max_value) or max_value == min_value:
        return pd.Series([0.0] * len(series), index=series.index)

    return (series - min_value) / (max_value - min_value)


# ---------------------------------------------------------
# Validation Helpers
# ---------------------------------------------------------
def validate_model_input(data, feature_columns=None):
    """
    Validate that the required model features and target column exist.
    This avoids silent model errors later in the pipeline.
    """

    if feature_columns is None:
        feature_columns = get_model_feature_columns(data)

    if not feature_columns:
        raise ValueError(
            "No valid model feature columns were found. "
            "Check preprocessing output and required AI4I columns."
        )

    missing_feature_columns = [
        col for col in feature_columns if col not in data.columns
    ]

    if missing_feature_columns:
        raise ValueError(
            "Missing required feature columns for model training: "
            + ", ".join(missing_feature_columns)
        )

    if TARGET_COLUMN not in data.columns:
        raise ValueError(f"Missing required target column: {TARGET_COLUMN}")

    if data[feature_columns].isnull().any().any():
        raise ValueError("Model input contains missing values in feature columns.")

    if data[TARGET_COLUMN].isnull().any():
        raise ValueError("Model input contains missing values in the target column.")

    target_values = data[TARGET_COLUMN].astype(int)

    if target_values.nunique() < 2:
        raise ValueError(
            "The target column must contain both normal and failure records "
            "for fair model evaluation."
        )


def get_train_test_split(scaled_data, feature_columns=None):
    """
    Create one consistent stratified train-test split for fair model comparison.
    Stratification is important because machine failure is an imbalanced target.
    """

    if feature_columns is None:
        feature_columns = get_model_feature_columns(scaled_data)

    validate_model_input(scaled_data, feature_columns)

    X = scaled_data[feature_columns]
    y = scaled_data[TARGET_COLUMN].astype(int)

    return train_test_split(
        X,
        y,
        test_size=TEST_SIZE,
        random_state=RANDOM_STATE,
        stratify=y,
    )


# ---------------------------------------------------------
# Evaluation Helpers
# ---------------------------------------------------------
def evaluate_model_predictions(model_name, y_true, y_pred, y_score=None):
    """
    Evaluate model predictions using metrics suitable for imbalanced failure data.

    accuracy  : overall correct predictions
    precision : how many predicted failures were real failures
    recall    : how many actual failures were detected
    f1_score  : balance between precision and recall
    pr_auc    : useful for imbalanced datasets
    roc_auc   : threshold-independent ranking metric
    """

    y_true = pd.Series(y_true).astype(int)
    y_pred = pd.Series(y_pred).astype(int)

    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()

    metrics = {
        "model": model_name,
        "accuracy": accuracy_score(y_true, y_pred),
        "precision": precision_score(y_true, y_pred, zero_division=0),
        "recall": recall_score(y_true, y_pred, zero_division=0),
        "f1_score": f1_score(y_true, y_pred, zero_division=0),
        "true_negative": int(tn),
        "false_positive": int(fp),
        "false_negative": int(fn),
        "true_positive": int(tp),
        "confusion_matrix": [[int(tn), int(fp)], [int(fn), int(tp)]],
    }

    if y_score is not None:
        try:
            metrics["roc_auc"] = roc_auc_score(y_true, y_score)
        except Exception:
            metrics["roc_auc"] = None

        try:
            metrics["pr_auc"] = average_precision_score(y_true, y_score)
        except Exception:
            metrics["pr_auc"] = None
    else:
        metrics["roc_auc"] = None
        metrics["pr_auc"] = None

    return metrics


def build_evaluation_dataframe(metrics_list):
    """
    Convert model metric dictionaries into a clean dataframe for evaluation_results.csv.
    """

    evaluation_df = pd.DataFrame(metrics_list)

    preferred_order = [
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
        "true_positive",
        "confusion_matrix",
        "oob_score",
    ]

    existing_order = [col for col in preferred_order if col in evaluation_df.columns]
    remaining_cols = [col for col in evaluation_df.columns if col not in existing_order]

    return evaluation_df[existing_order + remaining_cols]


# ---------------------------------------------------------
# Model 1: Isolation Forest
# ---------------------------------------------------------
def train_isolation_forest(
    scaled_data,
    feature_columns=None,
    X_train=None,
    y_train=None,
    X_test=None,
    y_test=None,
):
    """
    Train Isolation Forest as the main anomaly detection model.

    sklearn output:
    1  = normal
    -1 = anomaly

    Converted output:
    0 = normal
    1 = anomaly

    Important FYP logic:
    - For evaluation, the model is trained using normal training records only.
      This allows the model to learn normal operational behavior patterns.
    - For the final application output, the trained model predicts all records.
    """

    if feature_columns is None:
        feature_columns = get_model_feature_columns(scaled_data)

    if X_train is None or y_train is None or X_test is None or y_test is None:
        X_train, X_test, y_train, y_test = get_train_test_split(
            scaled_data,
            feature_columns,
        )

    normal_train_X = X_train[y_train == 0]

    if normal_train_X.empty:
        raise ValueError(
            "Isolation Forest cannot be trained because there are no normal "
            "records in the training data."
        )

    model = IsolationForest(
        n_estimators=200,
        contamination=ANOMALY_RATE,
        random_state=RANDOM_STATE,
        n_jobs=-1,
    )

    model.fit(normal_train_X)

    # Evaluation on the held-out test set
    test_predictions = model.predict(X_test)
    test_anomaly_labels = [1 if pred == -1 else 0 for pred in test_predictions]

    # decision_function: higher score = more normal
    # multiply by -1 to create operational risk score
    test_normality_scores = model.decision_function(X_test)
    test_risk_scores = -test_normality_scores

    metrics = evaluate_model_predictions(
        "Isolation Forest",
        y_test,
        test_anomaly_labels,
        y_score=test_risk_scores,
    )

    # Final monitoring predictions for the dashboard output
    X_full = scaled_data[feature_columns]
    full_predictions = model.predict(X_full)
    full_normality_scores = model.decision_function(X_full)
    full_anomaly_labels = [1 if pred == -1 else 0 for pred in full_predictions]

    result = scaled_data.copy()
    result["if_anomaly_label"] = full_anomaly_labels
    result["if_anomaly_score"] = full_normality_scores
    result["if_risk_score"] = -full_normality_scores

    test_results = X_test.copy()
    test_results["actual_failure"] = y_test.values
    test_results["if_prediction"] = test_anomaly_labels
    test_results["if_risk_score"] = test_risk_scores

    return model, metrics, result, test_results


# ---------------------------------------------------------
# Model 2: One-Class SVM
# ---------------------------------------------------------
def train_one_class_svm(
    scaled_data,
    feature_columns=None,
    X_train=None,
    y_train=None,
    X_test=None,
    y_test=None,
):
    """
    Train One-Class SVM as an unsupervised/novelty detection benchmark.

    sklearn output:
    1  = normal
    -1 = anomaly

    Converted output:
    0 = normal
    1 = anomaly
    """

    if feature_columns is None:
        feature_columns = get_model_feature_columns(scaled_data)

    if X_train is None or y_train is None or X_test is None or y_test is None:
        X_train, X_test, y_train, y_test = get_train_test_split(
            scaled_data,
            feature_columns,
        )

    normal_train_X = X_train[y_train == 0]

    if normal_train_X.empty:
        raise ValueError(
            "One-Class SVM cannot be trained because there are no normal "
            "records in the training data."
        )

    model = OneClassSVM(
        kernel="rbf",
        gamma="scale",
        nu=ANOMALY_RATE,
    )

    model.fit(normal_train_X)

    # Evaluation on the held-out test set
    test_predictions = model.predict(X_test)
    test_anomaly_labels = [1 if pred == -1 else 0 for pred in test_predictions]

    # decision_function: higher score = more normal
    # multiply by -1 to create operational risk score
    test_normality_scores = model.decision_function(X_test)
    test_risk_scores = -test_normality_scores

    metrics = evaluate_model_predictions(
        "One-Class SVM",
        y_test,
        test_anomaly_labels,
        y_score=test_risk_scores,
    )

    # Final monitoring predictions for the dashboard output
    X_full = scaled_data[feature_columns]
    full_predictions = model.predict(X_full)
    full_normality_scores = model.decision_function(X_full)
    full_anomaly_labels = [1 if pred == -1 else 0 for pred in full_predictions]

    result = scaled_data.copy()
    result["ocsvm_anomaly_label"] = full_anomaly_labels
    result["ocsvm_anomaly_score"] = full_normality_scores
    result["ocsvm_risk_score"] = -full_normality_scores

    test_results = X_test.copy()
    test_results["actual_failure"] = y_test.values
    test_results["ocsvm_prediction"] = test_anomaly_labels
    test_results["ocsvm_risk_score"] = test_risk_scores

    return model, metrics, result, test_results


# ---------------------------------------------------------
# Model 3: Random Forest Supervised Benchmark
# ---------------------------------------------------------
def train_random_forest(
    scaled_data,
    feature_columns=None,
    X_train=None,
    y_train=None,
    X_test=None,
    y_test=None,
):
    """
    Train Random Forest as a supervised benchmark model.

    Target variable:
    Machine failure classification

    Important FYP logic:
    - Random Forest is not the main anomaly detection model.
    - It is used as a supervised comparison model because labels are available in AI4I.
    - class_weight='balanced_subsample' helps with the imbalanced failure class.
    """

    if feature_columns is None:
        feature_columns = get_model_feature_columns(scaled_data)

    if X_train is None or y_train is None or X_test is None or y_test is None:
        X_train, X_test, y_train, y_test = get_train_test_split(
            scaled_data,
            feature_columns,
        )

    model = RandomForestClassifier(
        n_estimators=300,
        random_state=RANDOM_STATE,
        class_weight="balanced_subsample",
        min_samples_leaf=2,
        n_jobs=-1,
        bootstrap=True,
        oob_score=True,
    )

    model.fit(X_train, y_train)

    # Evaluation on the held-out test set
    y_pred = model.predict(X_test)
    y_proba = model.predict_proba(X_test)[:, 1]

    metrics = evaluate_model_predictions(
        "Random Forest",
        y_test,
        y_pred,
        y_score=y_proba,
    )

    metrics["oob_score"] = getattr(model, "oob_score_", None)

    test_results = X_test.copy()
    test_results["actual_failure"] = y_test.values
    test_results["rf_prediction"] = y_pred
    test_results["rf_failure_probability"] = y_proba

    # Final monitoring predictions for every machine/record
    X_full = scaled_data[feature_columns]
    full_predictions = model.predict(X_full)
    full_probabilities = model.predict_proba(X_full)[:, 1]

    result = scaled_data.copy()
    result["rf_prediction"] = full_predictions
    result["rf_failure_probability"] = full_probabilities

    feature_importance = pd.DataFrame(
        {
            "feature": feature_columns,
            "display_feature": [
                get_display_feature_name(feature) for feature in feature_columns
            ],
            "importance": model.feature_importances_,
        }
    ).sort_values("importance", ascending=False)

    return model, metrics, result, test_results, feature_importance


# ---------------------------------------------------------
# Final Risk Helper
# ---------------------------------------------------------
def add_combined_ai_risk(final_results):
    """
    Add a combined AI risk score using the three model outputs.

    This does not replace the individual model outputs.
    It provides a practical risk indicator for dashboard ranking and alert log creation.
    """

    result = final_results.copy()

    if "if_risk_score" in result.columns:
        result["if_risk_score_normalized"] = normalize_scores(result["if_risk_score"])
    else:
        result["if_risk_score_normalized"] = 0.0

    if "ocsvm_risk_score" in result.columns:
        result["ocsvm_risk_score_normalized"] = normalize_scores(
            result["ocsvm_risk_score"]
        )
    else:
        result["ocsvm_risk_score_normalized"] = 0.0

    if "rf_failure_probability" not in result.columns:
        result["rf_failure_probability"] = 0.0

    result["ai_anomaly_vote_count"] = (
        result.get("if_anomaly_label", 0).astype(int)
        + result.get("ocsvm_anomaly_label", 0).astype(int)
        + result.get("rf_prediction", 0).astype(int)
    )

    # Isolation Forest remains important because it is the main anomaly model.
    # Random Forest is strong because it uses the labelled failure target.
    result["combined_ai_risk_score"] = (
        0.40 * result["if_risk_score_normalized"]
        + 0.20 * result["ocsvm_risk_score_normalized"]
        + 0.40 * result["rf_failure_probability"].astype(float)
    )

    return result


# ---------------------------------------------------------
#  All Models
# ---------------------------------------------------------
def run_all_models(scaled_data):
    """
    Run all selected models:
    1. Isolation Forest      -> main anomaly detection model
    2. One-Class SVM         -> unsupervised anomaly detection benchmark
    3. Random Forest         -> supervised benchmark model

    Returns are kept backward-compatible with the previous pipeline.
    """

    feature_columns = get_model_feature_columns(scaled_data)
    validate_model_input(scaled_data, feature_columns)

    print("\nPreparing stratified train-test split for fair model evaluation...")
    print("Model feature columns used:")
    for feature in feature_columns:
        print(f"- {feature}")

    X_train, X_test, y_train, y_test = get_train_test_split(
        scaled_data,
        feature_columns,
    )

    print("\nTraining Isolation Forest...")
    isolation_model, if_metrics, if_results, if_test_results = train_isolation_forest(
        scaled_data,
        feature_columns=feature_columns,
        X_train=X_train,
        y_train=y_train,
        X_test=X_test,
        y_test=y_test,
    )
    print("Isolation Forest completed.")

    print("\nTraining One-Class SVM...")
    ocsvm_model, ocsvm_metrics, ocsvm_results, ocsvm_test_results = train_one_class_svm(
        scaled_data,
        feature_columns=feature_columns,
        X_train=X_train,
        y_train=y_train,
        X_test=X_test,
        y_test=y_test,
    )
    print("One-Class SVM completed.")

    print("\nTraining Random Forest...")
    (
        rf_model,
        rf_metrics,
        rf_results,
        rf_test_results,
        rf_feature_importance,
    ) = train_random_forest(
        scaled_data,
        feature_columns=feature_columns,
        X_train=X_train,
        y_train=y_train,
        X_test=X_test,
        y_test=y_test,
    )
    print("Random Forest completed.")

    final_results = scaled_data.copy()

    final_results["if_anomaly_label"] = if_results["if_anomaly_label"]
    final_results["if_anomaly_score"] = if_results["if_anomaly_score"]
    final_results["if_risk_score"] = if_results["if_risk_score"]

    final_results["ocsvm_anomaly_label"] = ocsvm_results["ocsvm_anomaly_label"]
    final_results["ocsvm_anomaly_score"] = ocsvm_results["ocsvm_anomaly_score"]
    final_results["ocsvm_risk_score"] = ocsvm_results["ocsvm_risk_score"]

    final_results["rf_prediction"] = rf_results["rf_prediction"]
    final_results["rf_failure_probability"] = rf_results["rf_failure_probability"]

    final_results = add_combined_ai_risk(final_results)

    evaluation_results = build_evaluation_dataframe(
        [
            if_metrics,
            ocsvm_metrics,
            rf_metrics,
        ]
    )

    print("\nModel evaluation summary:")
    summary_columns = [
        "model",
        "accuracy",
        "precision",
        "recall",
        "f1_score",
        "pr_auc",
    ]
    existing_summary_columns = [
        col for col in summary_columns if col in evaluation_results.columns
    ]
    print(evaluation_results[existing_summary_columns])

    return {
        "isolation_model": isolation_model,
        "ocsvm_model": ocsvm_model,
        "rf_model": rf_model,

        # Backward-compatible keys
        "rf_metrics": rf_metrics,
        "rf_test_results": rf_test_results,

        # Improved model evaluation outputs
        "if_metrics": if_metrics,
        "ocsvm_metrics": ocsvm_metrics,
        "evaluation_results": evaluation_results,
        "if_test_results": if_test_results,
        "ocsvm_test_results": ocsvm_test_results,
        "rf_feature_importance": rf_feature_importance,

        # Feature information used by the model pipeline
        "model_feature_columns": feature_columns,

        # Final monitoring dataset output for dashboard and reporting
        "final_results": final_results,
    }
