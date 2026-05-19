from src.load_data import load_dataset
from src.preprocessing import preprocess_data, save_preprocessed_data
from src.model_training import run_all_models
from src.evaluation import save_evaluation_results
from src.decision_support import apply_decision_support, save_final_results
from src.machine_simulation import assign_machines_to_records, save_machine_results
from src.explainability import apply_explainability


def main():
    dataset_path = "data/ai4i2020.csv"

    # ---------------------------------------------------------
    # Step 1: Load dataset
    # ---------------------------------------------------------
    data = load_dataset(dataset_path)

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

    final_results = model_outputs["final_results"]
    evaluation_results = model_outputs["evaluation_results"]

    if_test_results = model_outputs["if_test_results"]
    ocsvm_test_results = model_outputs["ocsvm_test_results"]
    rf_test_results = model_outputs["rf_test_results"]
    rf_feature_importance = model_outputs["rf_feature_importance"]

    # ---------------------------------------------------------
    # Step 5: Save model outputs
    # ---------------------------------------------------------
    final_results.to_csv("outputs/model_results.csv", index=False)
    evaluation_results.to_csv("outputs/evaluation_results.csv", index=False)

    if_test_results.to_csv("outputs/isolation_forest_test_results.csv", index=False)
    ocsvm_test_results.to_csv("outputs/one_class_svm_test_results.csv", index=False)
    rf_test_results.to_csv("outputs/random_forest_test_results.csv", index=False)
    rf_feature_importance.to_csv("outputs/random_forest_feature_importance.csv", index=False)

    print("\nModel results saved to: outputs/model_results.csv")
    print("Evaluation results saved to: outputs/evaluation_results.csv")
    print("Isolation Forest test results saved to: outputs/isolation_forest_test_results.csv")
    print("One-Class SVM test results saved to: outputs/one_class_svm_test_results.csv")
    print("Random Forest test results saved to: outputs/random_forest_test_results.csv")
    print("Random Forest feature importance saved to: outputs/random_forest_feature_importance.csv")

    # Save evaluation results
    save_evaluation_results(evaluation_results)

    print("\nEvaluation Results:")
    print(evaluation_results)

    # ---------------------------------------------------------
    # Step 6: Generate AI-based health score and alert levels
    # ---------------------------------------------------------
    final_machine_results = apply_decision_support(final_results)
    save_final_results(final_machine_results)

    # ---------------------------------------------------------
    # Step 7: Assign simulated equipment categories and machine IDs
    # ---------------------------------------------------------
    machine_results = assign_machines_to_records(final_machine_results)

    # ---------------------------------------------------------
    # Step 8: Add rule-based anomaly explanations
    # and maintenance recommendations
    # ---------------------------------------------------------
    machine_results = apply_explainability(machine_results)

    # ---------------------------------------------------------
    # Step 9: Save final monitoring results
    # ---------------------------------------------------------
    save_machine_results(machine_results)

    print("\nSample machine monitoring results:")

    preview_columns = [
        "machine_id",
        "equipment_category",
        "Machine failure",
        "if_anomaly_label",
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

    print("\nAI-Based Predictive Maintenance Pipeline completed successfully.")


if __name__ == "__main__":
    main()