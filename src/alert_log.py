import os
from datetime import datetime

import pandas as pd


ALERT_SEVERITY_ORDER = {
    "Normal": 0,
    "Low Risk": 1,
    "Medium Risk": 2,
    "High Risk": 3,
    "Critical": 4,
}


ALERT_STATUS_BY_LEVEL = {
    "Low Risk": "Active",
    "Medium Risk": "Active",
    "High Risk": "Active",
    "Critical": "Active",
}


def _classify_machine_alert(machine_health_score, worst_alert_severity, anomaly_rate):
    """
    Classify final machine-level alert using aggregated health and repeated
    anomaly evidence. This avoids judging a full machine from one isolated row.
    """

    if machine_health_score >= 80:
        alert_level = "Normal"
    elif machine_health_score >= 60:
        alert_level = "Low Risk"
    elif machine_health_score >= 50:
        alert_level = "Medium Risk"
    elif machine_health_score >= 40:
        alert_level = "High Risk"
    else:
        alert_level = "Critical"

    if worst_alert_severity >= ALERT_SEVERITY_ORDER["Critical"] and anomaly_rate >= 0.06 and machine_health_score < 58:
        alert_level = "Critical"
    elif worst_alert_severity >= ALERT_SEVERITY_ORDER["High Risk"] and anomaly_rate >= 0.08 and machine_health_score < 62:
        if ALERT_SEVERITY_ORDER.get(alert_level, 0) < ALERT_SEVERITY_ORDER["High Risk"]:
            alert_level = "High Risk"
    elif worst_alert_severity >= ALERT_SEVERITY_ORDER["Medium Risk"] and anomaly_rate >= 0.05 and machine_health_score < 66:
        if ALERT_SEVERITY_ORDER.get(alert_level, 0) < ALERT_SEVERITY_ORDER["Medium Risk"]:
            alert_level = "Medium Risk"

    return alert_level


def _classify_alert_priority(alert_level):
    priority_map = {
        "Normal": "Stable",
        "Low Risk": "Observe",
        "Medium Risk": "Inspect",
        "High Risk": "Urgent",
        "Critical": "Immediate",
    }
    return priority_map.get(str(alert_level), "Unknown")


def generate_maintenance_alert_log(machine_results):
    """
    Generate a professional one-row-per-machine maintenance alert log.

    The full monitoring output contains many records per simulated machine.
    For the alert log, the system aggregates each machine first, then keeps only
    machines that are not Normal.
    """

    if machine_results is None or machine_results.empty:
        return pd.DataFrame()

    df = machine_results.copy()

    required_defaults = {
        "machine_id": "N/A",
        "equipment_category": "General Equipment",
        "alert_level": "Normal",
        "alert_priority": "Stable",
        "health_score": 100,
        "if_anomaly_label": 0,
        "problem_detected": "No significant abnormality detected",
        "probable_cause": "Machine condition appears stable",
        "recommended_action": "Continue normal operation and routine monitoring",
        "short_reason": "No significant abnormality detected",
    }

    for column, default_value in required_defaults.items():
        if column not in df.columns:
            df[column] = default_value

    df["_alert_severity"] = df["alert_level"].map(ALERT_SEVERITY_ORDER).fillna(0)
    df["health_score"] = pd.to_numeric(df["health_score"], errors="coerce").fillna(100)
    df["if_anomaly_label"] = pd.to_numeric(df["if_anomaly_label"], errors="coerce").fillna(0)

    machine_rows = []

    for machine_id, group in df.groupby("machine_id", sort=False):
        group = group.copy()

        avg_health = float(group["health_score"].mean())
        worst_health = float(group["health_score"].min())
        anomaly_rate = float(group["if_anomaly_label"].mean())
        worst_alert_severity = int(group["_alert_severity"].max())

        machine_health_score = round(
            (avg_health * 0.55) +
            (worst_health * 0.30) +
            ((100 - (anomaly_rate * 100)) * 0.15),
            2,
        )

        representative_row = group.sort_values(
            ["_alert_severity", "health_score"],
            ascending=[False, True],
        ).iloc[0].copy()

        final_alert_level = _classify_machine_alert(
            machine_health_score,
            worst_alert_severity,
            anomaly_rate,
        )

        representative_row["health_score"] = machine_health_score
        representative_row["alert_level"] = final_alert_level
        representative_row["alert_priority"] = _classify_alert_priority(final_alert_level)
        representative_row["_alert_severity"] = ALERT_SEVERITY_ORDER.get(final_alert_level, 0)
        representative_row["_anomaly_rate"] = round(anomaly_rate, 4)

        machine_rows.append(representative_row)

    if not machine_rows:
        return pd.DataFrame()

    alert_df = pd.DataFrame(machine_rows)
    alert_df = alert_df[alert_df["alert_level"].astype(str) != "Normal"].copy()

    if alert_df.empty:
        return pd.DataFrame(
            columns=[
                "alert_id",
                "machine_id",
                "equipment_category",
                "alert_level",
                "alert_priority",
                "health_score",
                "problem",
                "probable_cause",
                "recommended_action",
                "observed_pattern",
                "alert_status",
                "source_model",
                "generated_at",
            ]
        )

    alert_df = alert_df.sort_values(
        ["_alert_severity", "health_score"],
        ascending=[False, True],
    ).reset_index(drop=True)

    generated_at = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    output_df = pd.DataFrame(
        {
            "alert_id": [f"ALT-{index + 1:04d}" for index in range(len(alert_df))],
            "machine_id": alert_df["machine_id"].astype(str),
            "equipment_category": alert_df["equipment_category"].astype(str),
            "alert_level": alert_df["alert_level"].astype(str),
            "alert_priority": alert_df["alert_priority"].astype(str),
            "health_score": alert_df["health_score"].round(2),
            "problem": alert_df["problem_detected"].astype(str),
            "probable_cause": alert_df["probable_cause"].astype(str),
            "recommended_action": alert_df["recommended_action"].astype(str),
            "observed_pattern": alert_df["short_reason"].astype(str),
            "alert_status": alert_df["alert_level"].map(ALERT_STATUS_BY_LEVEL).fillna("Active"),
            "source_model": "Isolation Forest + Rule-Based Decision Support",
            "generated_at": generated_at,
        }
    )

    return output_df

def save_maintenance_alert_log(alert_log, output_folder="outputs"):
    """
    Save the maintenance alert log as a CSV file for dashboard/report usage.
    """

    os.makedirs(output_folder, exist_ok=True)
    output_path = os.path.join(output_folder, "maintenance_alert_log.csv")
    alert_log.to_csv(output_path, index=False)
    print(f"Maintenance alert log saved to: {output_path}")
