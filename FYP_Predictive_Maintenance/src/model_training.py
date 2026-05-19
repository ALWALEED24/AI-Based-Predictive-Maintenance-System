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


FEATURE_COLUMNS = [
    "Type",
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


# ---------------------------------------------------------
# Validation Helpers
# ---------------------------------------------------------
def validate_model_input(data):
    """
    Validate that the required model features and target column exist.
    This avoids silent model errors later in the pipeline.
    """

    required_columns = FEATURE_COLUMNS + [TARGET_COLUMN]
    missing_columns = [col for col in required_columns if col not in data.columns]

    if missing_columns:
        raise ValueError(
            "Missing required columns for model training: " + ", ".join(missing_columns)
        )

    if data[FEATURE_COLUMNS].isnull().any().any():
        raise ValueError("Model input contains missing values in feature columns.")

    if data[TARGET_COLUMN].isnull().any():
        raise ValueError("Model input contains missing values in the target column.")


def get_train_test_split(scaled_data):
    """
    Create one consistent stratified train-test split for fair model comparison.
    Stratification is important because machine failure is an imbalanced target.
    """

    validate_model_input(scaled_data)

    X = scaled_data[FEATURE_COLUMNS]
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

    if X_train is None or y_train is None or X_test is None or y_test is None:
        X_train, X_test, y_train, y_test = get_train_test_split(scaled_data)

    normal_train_X = X_train[y_train == 0]

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
    X_full = scaled_data[FEATURE_COLUMNS]
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

    if X_train is None or y_train is None or X_test is None or y_test is None:
        X_train, X_test, y_train, y_test = get_train_test_split(scaled_data)

    normal_train_X = X_train[y_train == 0]

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
    X_full = scaled_data[FEATURE_COLUMNS]
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

    if X_train is None or y_train is None or X_test is None or y_test is None:
        X_train, X_test, y_train, y_test = get_train_test_split(scaled_data)

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

    feature_importance = pd.DataFrame(
        {
            "feature": FEATURE_COLUMNS,
            "importance": model.feature_importances_,
        }
    ).sort_values("importance", ascending=False)

    return model, metrics, test_results, feature_importance


# ---------------------------------------------------------
# Run All Models
# ---------------------------------------------------------
def run_all_models(scaled_data):
    """
    Run all selected models:
    1. Isolation Forest      -> main anomaly detection model
    2. One-Class SVM         -> unsupervised anomaly detection benchmark
    3. Random Forest         -> supervised benchmark model

    Returns are kept backward-compatible with the previous pipeline.
    """

    print("\nPreparing stratified train-test split for fair model evaluation...")
    X_train, X_test, y_train, y_test = get_train_test_split(scaled_data)

    print("\nTraining Isolation Forest...")
    isolation_model, if_metrics, if_results, if_test_results = train_isolation_forest(
        scaled_data,
        X_train=X_train,
        y_train=y_train,
        X_test=X_test,
        y_test=y_test,
    )
    print("Isolation Forest completed.")

    print("\nTraining One-Class SVM...")
    ocsvm_model, ocsvm_metrics, ocsvm_results, ocsvm_test_results = train_one_class_svm(
        scaled_data,
        X_train=X_train,
        y_train=y_train,
        X_test=X_test,
        y_test=y_test,
    )
    print("One-Class SVM completed.")

    print("\nTraining Random Forest...")
    rf_model, rf_metrics, rf_test_results, rf_feature_importance = train_random_forest(
        scaled_data,
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

    evaluation_results = build_evaluation_dataframe(
        [
            if_metrics,
            ocsvm_metrics,
            rf_metrics,
        ]
    )

    print("\nModel evaluation summary:")
    print(
        evaluation_results[
            [
                "model",
                "accuracy",
                "precision",
                "recall",
                "f1_score",
                "pr_auc",
            ]
        ]
    )

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

        # Final monitoring dataset output for dashboard and reporting
        "final_results": final_results,
    }