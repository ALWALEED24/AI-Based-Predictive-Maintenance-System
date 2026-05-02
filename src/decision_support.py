import os
from sklearn.preprocessing import MinMaxScaler


def generate_health_score(model_results):
    """
    Generate health score from Isolation Forest anomaly score.
    Higher score = healthier machine condition.
    Lower score = more abnormal condition.
    """

    df = model_results.copy()

    scaler = MinMaxScaler(feature_range=(0, 100))

    df["health_score"] = scaler.fit_transform(
        df[["if_anomaly_score"]]
    ).round(2)

    return df


def classify_alert_level(health_score):
    """
    Convert health score into alert level.
    """

    if health_score >= 80:
        return "Normal"
    elif health_score >= 60:
        return "Low Risk"
    elif health_score >= 40:
        return "Medium Risk"
    elif health_score >= 20:
        return "High Risk"
    else:
        return "Critical"


def classify_alert_priority(alert_level):
    """
    Convert alert level into alert priority.
    """

    priority_map = {
        "Normal": "Stable",
        "Low Risk": "Observe",
        "Medium Risk": "Inspect",
        "High Risk": "Urgent",
        "Critical": "Immediate"
    }

    return priority_map.get(alert_level, "Unknown")


def generate_recommendation(alert_level):
    """
    Generate maintenance recommendation based on alert level.
    """

    recommendations = {
        "Normal": "No immediate action required. Continue normal monitoring.",
        "Low Risk": "Increase monitoring frequency and review machine condition during the next inspection.",
        "Medium Risk": "Schedule machine inspection and check possible abnormal operating conditions.",
        "High Risk": "Plan maintenance soon and reduce machine load if possible.",
        "Critical": "Immediate maintenance action required. Stop or isolate the machine if necessary."
    }

    return recommendations.get(alert_level, "No recommendation available.")


def apply_decision_support(model_results):
    """
    Apply health score, alert level, priority, and recommendation.
    """

    df = generate_health_score(model_results)

    df["alert_level"] = df["health_score"].apply(classify_alert_level)
    df["alert_priority"] = df["alert_level"].apply(classify_alert_priority)
    df["recommended_action"] = df["alert_level"].apply(generate_recommendation)

    print("\nDecision support output generated successfully.")
    print("Health score, alert level, priority, and recommendation added.")

    return df


def save_final_results(final_machine_results, output_folder="outputs"):
    """
    Save final machine results.
    """

    os.makedirs(output_folder, exist_ok=True)

    output_path = os.path.join(output_folder, "final_machine_results.csv")
    final_machine_results.to_csv(output_path, index=False)

    print(f"\nFinal machine results saved to: {output_path}")