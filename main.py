import os
import pandas as pd

from src.load_data import load_dataset
from src.preprocessing import preprocess_data, save_preprocessed_data
from src.model_training import run_all_models
from src.evaluation import save_evaluation_results
from src.decision_support import apply_decision_support, save_final_results
from src.machine_simulation import assign_machines_to_records, save_machine_results
from src.explainability import apply_explainability

try:
    from src.alert_log import generate_maintenance_alert_log, save_maintenance_alert_log
except ImportError:
    generate_maintenance_alert_log = None
    save_maintenance_alert_log = None


DATASET_PATH = "data/ai4i2020.csv"
OUTPUT_DIR = "outputs"

DISPLAY_SENSOR_COLUMNS = [
    "Type",
    "Air temperature [K]",
    "Process temperature [K]",
    "Rotational speed [rpm]",
    "Torque [Nm]",
    "Tool wear [min]",
]

TARGET_COLUMNS = [
    "Machine failure",
    "TWF",
    "HDF",
    "PWF",
    "OSF",
    "RNF",
]


def ensure_output_directory():
    """Make sure the outputs folder exists before saving CSV files."""
    os.makedirs(OUTPUT_DIR, exist_ok=True)


def restore_original_display_values(model_results, cleaned_data, scaled_data):
    """
    Keep scaled values for AI logic, but restore original readable values for
    dashboard/report display.

    Why this is needed:
    - ML models should use scaled values.
    - The dashboard and reports should show real values such as 298 K, 1550 rpm,
      40 Nm, and 120 min.
    - The rule-based explainability layer can still use scaled_* columns.
    """

    final_results = model_results.copy().reset_index(drop=True)
    cleaned_copy = cleaned_data.copy().reset_index(drop=True)
    scaled_copy = scaled_data.copy().reset_index(drop=True)

    # Store scaled model input values using scaled_* names.
    for column in DISPLAY_SENSOR_COLUMNS:
        if column in scaled_copy.columns:
            final_results[f"scaled_{column}"] = scaled_copy[column].values

    # Restore original readable operational values for display and reports.
    for column in DISPLAY_SENSOR_COLUMNS:
        if column in cleaned_copy.columns:
            final_results[column] = cleaned_copy[column].values

    # Keep target/failure labels aligned with the original cleaned dataset.
    for column in TARGET_COLUMNS:
        if column in cleaned_copy.columns:
            final_results[column] = cleaned_copy[column].values

    return final_results


def save_model_outputs(model_outputs, final_results):
    """Save model outputs and evaluation files."""
    evaluation_results = model_outputs["evaluation_results"]
    if_test_results = model_outputs["if_test_results"]
    ocsvm_test_results = model_outputs["ocsvm_test_results"]
    rf_test_results = model_outputs["rf_test_results"]
    rf_feature_importance = model_outputs["rf_feature_importance"]

    final_results.to_csv(os.path.join(OUTPUT_DIR, "model_results.csv"), index=False)
    evaluation_results.to_csv(os.path.join(OUTPUT_DIR, "evaluation_results.csv"), index=False)

    if_test_results.to_csv(os.path.join(OUTPUT_DIR, "isolation_forest_test_results.csv"), index=False)
    ocsvm_test_results.to_csv(os.path.join(OUTPUT_DIR, "one_class_svm_test_results.csv"), index=False)
    rf_test_results.to_csv(os.path.join(OUTPUT_DIR, "random_forest_test_results.csv"), index=False)
    rf_feature_importance.to_csv(os.path.join(OUTPUT_DIR, "random_forest_feature_importance.csv"), index=False)

    print("\nModel results saved to: outputs/model_results.csv")
    print("Evaluation results saved to: outputs/evaluation_results.csv")
    print("Isolation Forest test results saved to: outputs/isolation_forest_test_results.csv")
    print("One-Class SVM test results saved to: outputs/one_class_svm_test_results.csv")
    print("Random Forest test results saved to: outputs/random_forest_test_results.csv")
    print("Random Forest feature importance saved to: outputs/random_forest_feature_importance.csv")

    save_evaluation_results(evaluation_results)

    print("\nEvaluation Results:")
    print(evaluation_results)

    return evaluation_results


def build_alert_log_if_available(machine_results):
    """
    Generate the maintenance alert log if src/alert_log.py exists.
    This keeps main.py safe even if the alert log module has not been added yet.
    """

    if generate_maintenance_alert_log is None or save_maintenance_alert_log is None:
        print("\nMaintenance alert log module not found. Skipping alert log generation.")
        return None

    alert_log = generate_maintenance_alert_log(machine_results)
    save_maintenance_alert_log(alert_log)

    print("\nMaintenance alert log saved to: outputs/maintenance_alert_log.csv")
    return alert_log


def print_preview(machine_results):
    """Print a small preview of the final monitoring output."""
    print("\nSample machine monitoring results:")

    preview_columns = [
        "machine_id",
        "equipment_category",
        "Machine failure",
        "if_anomaly_label",
        "ocsvm_anomaly_label",
        "rf_prediction",
        "rf_failure_probability",
        "combined_ai_risk_score",
        "health_score",
        "alert_level",
        "alert_priority",
        "short_reason",
        "recommended_action",
        "explanation",
    ]

    existing_preview_columns = [
        col for col in preview_columns if col in machine_results.columns
    ]

    print(machine_results[existing_preview_columns].head(12))


def main():
    ensure_output_directory()

    # ---------------------------------------------------------
    # Step 1: Load dataset
    # ---------------------------------------------------------
    data = load_dataset(DATASET_PATH)

    if data is None:
        print("Program stopped because dataset could not be loaded.")
        return

    # ---------------------------------------------------------
    # Step 2: Preprocess operational sensor dataset
    # ---------------------------------------------------------
    cleaned_data, scaled_data, scaler = preprocess_data(data)

    if cleaned_data is None or scaled_data is None:
        print("Program stopped because preprocessing failed.")
        return

    # ---------------------------------------------------------
    # Step 3: Save preprocessed outputs
    # ---------------------------------------------------------
    save_preprocessed_data(cleaned_data, scaled_data)

    # ---------------------------------------------------------
    # Step 4: Run AI anomaly detection and ML models
    # ---------------------------------------------------------
    model_outputs = run_all_models(scaled_data)

    # Raw model output is based on scaled data.
    model_results = model_outputs["final_results"]

    # Restore original readable sensor values while keeping scaled_* columns.
    final_results = restore_original_display_values(
        model_results=model_results,
        cleaned_data=cleaned_data,
        scaled_data=scaled_data,
    )

    # ---------------------------------------------------------
    # Step 5: Save model outputs
    # ---------------------------------------------------------
    save_model_outputs(model_outputs, final_results)

    # ---------------------------------------------------------
    # Step 6: Assign simulated equipment categories and machine IDs
    # ---------------------------------------------------------
    # Important:
    # Equipment categories must be assigned before decision support because
    # alert level rules and maintenance recommendations depend on equipment type.
    machine_results = assign_machines_to_records(final_results)

    # ---------------------------------------------------------
    # Step 7: Generate AI-based health score and alert levels
    # ---------------------------------------------------------
    machine_results = apply_decision_support(machine_results)
    save_final_results(machine_results)

    # ---------------------------------------------------------
    # Step 8: Add rule-based anomaly explanations
    # and maintenance recommendations
    # ---------------------------------------------------------
    machine_results = apply_explainability(machine_results)

    # ---------------------------------------------------------
    # Step 9: Generate maintenance alert log
    # ---------------------------------------------------------
    build_alert_log_if_available(machine_results)

    # ---------------------------------------------------------
    # Step 10: Save final monitoring results
    # ---------------------------------------------------------
    save_machine_results(machine_results)

    print_preview(machine_results)

    print("\nAI-Based Predictive Maintenance Pipeline completed successfully.")


if __name__ == "__main__":
    main()
