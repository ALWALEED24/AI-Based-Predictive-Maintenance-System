from src.load_data import load_dataset
from src.preprocessing import preprocess_data, save_preprocessed_data
from src.model_training import run_all_models
from src.evaluation import evaluate_all_models, save_evaluation_results
from src.decision_support import apply_decision_support, save_final_results
from src.machine_simulation import assign_machines_to_records, save_machine_results
from src.explainability import apply_explainability


def main():
    dataset_path = "data/ai4i2020.csv"

    # Step 1: Load dataset
    data = load_dataset(dataset_path)

    if data is None:
        print("Program stopped because dataset could not be loaded.")
        return

    # Step 2: Preprocess dataset
    cleaned_data, scaled_data, scaler = preprocess_data(data)

    if cleaned_data is None or scaled_data is None:
        print("Program stopped because preprocessing failed.")
        return

    # Step 3: Save preprocessed output files
    save_preprocessed_data(cleaned_data, scaled_data)

    # Step 4: Train models
    model_outputs = run_all_models(scaled_data)

    final_results = model_outputs["final_results"]
    rf_metrics = model_outputs["rf_metrics"]
    rf_test_results = model_outputs["rf_test_results"]

    # Step 5: Save model outputs
    final_results.to_csv("outputs/model_results.csv", index=False)
    rf_test_results.to_csv("outputs/random_forest_test_results.csv", index=False)

    print("\nModel results saved to: outputs/model_results.csv")
    print("Random Forest test results saved to: outputs/random_forest_test_results.csv")

    # Step 6: Evaluate all models
    evaluation_df = evaluate_all_models(final_results, rf_metrics)
    save_evaluation_results(evaluation_df)

    print("\nEvaluation Results:")
    print(evaluation_df)

    # Step 7: Generate health score and alert level
    final_machine_results = apply_decision_support(final_results)
    save_final_results(final_machine_results)

    # Step 8: Assign simulated machine IDs and equipment categories
    machine_results = assign_machines_to_records(final_machine_results)

    # Step 9: Add realistic explanation and equipment-specific recommendations
    machine_results = apply_explainability(machine_results)

    # Step 10: Save final machine results
    save_machine_results(machine_results)

    print("\nSample machine results with realistic recommendations:")
    print(
        machine_results[
            [
                "machine_id",
                "equipment_category",
                "Machine failure",
                "if_anomaly_label",
                "health_score",
                "alert_level",
                "alert_priority",
                "short_reason",
                "recommended_action",
                "explanation"
            ]
        ].head(12)
    )


if __name__ == "__main__":
    main()