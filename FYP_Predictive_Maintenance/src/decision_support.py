import os
from sklearn.preprocessing import MinMaxScaler


# ---------------------------------------------------------
# Equipment-Specific Alert Thresholds
# ---------------------------------------------------------
EQUIPMENT_ALERT_THRESHOLDS = {
    "Motor": {
        "normal": 80,
        "low": 60,
        "medium": 40,
        "high": 20
    },
    "Pump": {
        "normal": 82,
        "low": 62,
        "medium": 42,
        "high": 22
    },
    "Compressor": {
        "normal": 85,
        "low": 65,
        "medium": 45,
        "high": 25
    },
    "Gearbox": {
        "normal": 78,
        "low": 58,
        "medium": 38,
        "high": 18
    },
    "Conveyor": {
        "normal": 76,
        "low": 56,
        "medium": 36,
        "high": 18
    },
    "Fan": {
        "normal": 75,
        "low": 55,
        "medium": 35,
        "high": 15
    },
    "Blower": {
        "normal": 75,
        "low": 55,
        "medium": 35,
        "high": 15
    },
    "Hydraulic System": {
        "normal": 82,
        "low": 62,
        "medium": 42,
        "high": 22
    },
    "Turbine": {
        "normal": 88,
        "low": 68,
        "medium": 48,
        "high": 28
    },
    "Generator": {
        "normal": 86,
        "low": 66,
        "medium": 46,
        "high": 26
    },
    "Chiller": {
        "normal": 80,
        "low": 60,
        "medium": 40,
        "high": 20
    },
    "Heat Exchanger": {
        "normal": 78,
        "low": 58,
        "medium": 38,
        "high": 18
    },
    "Cooling Tower": {
        "normal": 76,
        "low": 56,
        "medium": 36,
        "high": 16
    },
    "Centrifuge": {
        "normal": 86,
        "low": 66,
        "medium": 46,
        "high": 26
    },
    "Mixer": {
        "normal": 78,
        "low": 58,
        "medium": 38,
        "high": 18
    },
    "Agitator": {
        "normal": 77,
        "low": 57,
        "medium": 37,
        "high": 17
    },
    "Vacuum Pump": {
        "normal": 84,
        "low": 64,
        "medium": 44,
        "high": 24
    },
    "Separator": {
        "normal": 82,
        "low": 62,
        "medium": 42,
        "high": 22
    },
    "General Equipment": {
        "normal": 80,
        "low": 60,
        "medium": 40,
        "high": 20
    }
}


def generate_health_score(model_results):
    """
    Generate an AI-derived operational risk health score
    from the Isolation Forest anomaly score.

    Higher score = healthier operating pattern.
    Lower score = more abnormal operating pattern.

    Note:
    This score is a relative condition indicator for decision support,
    not a direct physical degradation measurement.
    """

    df = model_results.copy()

    scaler = MinMaxScaler(feature_range=(0, 100))

    df["health_score"] = scaler.fit_transform(
        df[["if_anomaly_score"]]
    ).round(2)

    return df


def get_equipment_thresholds(equipment_category):
    """
    Return equipment-specific alert thresholds.
    If the equipment category is not available, use general thresholds.
    """

    equipment_category = str(equipment_category)

    return EQUIPMENT_ALERT_THRESHOLDS.get(
        equipment_category,
        EQUIPMENT_ALERT_THRESHOLDS["General Equipment"]
    )


def classify_alert_level(health_score, equipment_category="General Equipment"):
    """
    Convert health score into alert level using equipment-specific thresholds.

    This means different equipment types can have different sensitivity.
    For example, a Turbine or Compressor can become risky earlier than a Fan
    because critical equipment usually needs stricter monitoring.
    """

    thresholds = get_equipment_thresholds(equipment_category)

    if health_score >= thresholds["normal"]:
        return "Normal"
    elif health_score >= thresholds["low"]:
        return "Low Risk"
    elif health_score >= thresholds["medium"]:
        return "Medium Risk"
    elif health_score >= thresholds["high"]:
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
    Generate condition-based maintenance recommendation
    based on alert level.
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
    Apply operational health scoring, equipment-specific alert level classification,
    alert priority, and condition-based maintenance recommendation.
    """

    df = generate_health_score(model_results)

    if "equipment_category" not in df.columns:
        df["equipment_category"] = "General Equipment"

    df["alert_level"] = df.apply(
        lambda row: classify_alert_level(
            row["health_score"],
            row["equipment_category"]
        ),
        axis=1
    )

    df["alert_priority"] = df["alert_level"].apply(classify_alert_priority)
    df["recommended_action"] = df["alert_level"].apply(generate_recommendation)

    print("\nDecision support output generated successfully.")
    print("Operational health score, equipment-specific alert level, priority, and maintenance recommendation added.")

    return df


def save_final_results(final_machine_results, output_folder="outputs"):
    """
    Save final decision-support results.
    """

    os.makedirs(output_folder, exist_ok=True)

    output_path = os.path.join(output_folder, "final_machine_results.csv")
    final_machine_results.to_csv(output_path, index=False)

    print(f"\nFinal decision-support results saved to: {output_path}")