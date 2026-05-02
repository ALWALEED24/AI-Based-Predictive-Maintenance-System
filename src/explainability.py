def get_equipment_context(equipment_category):
    """
    Equipment-specific maintenance knowledge.
    This is used as a rule-based decision-support layer after ML output.
    """

    context = {
        "Motor": {
            "normal": {
                "problem": "No significant motor abnormality detected",
                "cause": "Motor readings are within an acceptable operating pattern",
                "solution": "Continue normal operation and routine monitoring"
            },
            "low": {
                "problem": "Early motor load variation",
                "cause": "Slight change in torque, temperature, or rotational behaviour",
                "solution": "Increase monitoring frequency and check motor ventilation during the next inspection"
            },
            "medium": {
                "problem": "Possible motor overload or heating issue",
                "cause": "Abnormal torque or temperature pattern may indicate load stress or cooling limitation",
                "solution": "Inspect motor temperature, current draw, coupling alignment, and cooling path"
            },
            "high": {
                "problem": "High motor stress condition",
                "cause": "Low health score may indicate overload, insulation stress, or abnormal torque demand",
                "solution": "Reduce motor load if possible and schedule maintenance soon"
            },
            "critical": {
                "problem": "Critical motor operating condition",
                "cause": "Severe abnormal pattern may indicate overheating, overload, or potential motor failure",
                "solution": "Stop or isolate the motor if safety risk exists and perform urgent inspection"
            }
        },

        "Pump": {
            "normal": {
                "problem": "No significant pump abnormality detected",
                "cause": "Pump operating pattern appears stable",
                "solution": "Continue normal monitoring and routine pump checks"
            },
            "low": {
                "problem": "Early pump performance deviation",
                "cause": "Minor change in load, speed, or temperature may indicate early flow disturbance",
                "solution": "Monitor flow condition and check for small pressure or suction changes"
            },
            "medium": {
                "problem": "Possible cavitation or pump load issue",
                "cause": "Abnormal torque or speed pattern may suggest suction restriction, impeller blockage, or bearing wear",
                "solution": "Inspect suction pressure, impeller condition, seals, and bearing vibration"
            },
            "high": {
                "problem": "High-risk pump operating condition",
                "cause": "Low health score may indicate cavitation, bearing wear, or unstable hydraulic load",
                "solution": "Plan maintenance soon and reduce pump load if possible"
            },
            "critical": {
                "problem": "Critical pump condition",
                "cause": "Severe abnormal pattern may indicate cavitation damage, bearing failure, or major flow blockage",
                "solution": "Stop pump if necessary and urgently inspect suction line, impeller, seals, and bearings"
            }
        },

        "Compressor": {
            "normal": {
                "problem": "No significant compressor abnormality detected",
                "cause": "Compressor readings are stable",
                "solution": "Continue regular monitoring"
            },
            "low": {
                "problem": "Early compressor efficiency variation",
                "cause": "Minor temperature or load fluctuation may indicate early operating instability",
                "solution": "Monitor discharge temperature, pressure, and oil condition"
            },
            "medium": {
                "problem": "Possible compressor overheating or pressure instability",
                "cause": "Abnormal temperature, torque, or speed pattern may suggest lubrication issue or unstable compression load",
                "solution": "Inspect oil level, filters, discharge pressure, temperature, and vibration"
            },
            "high": {
                "problem": "High compressor stress condition",
                "cause": "Low health score may indicate overheating, overpressure, or lubrication problem",
                "solution": "Schedule maintenance soon and check whether the compressor can continue operation safely"
            },
            "critical": {
                "problem": "Critical compressor condition",
                "cause": "Severe abnormal pattern may indicate major pressure instability, overheating, or lubrication failure",
                "solution": "Stop or isolate compressor if required and perform urgent technical inspection"
            }
        },

        "Fan": {
            "normal": {
                "problem": "No significant fan abnormality detected",
                "cause": "Fan speed, load, and temperature pattern appear stable",
                "solution": "Continue normal fan operation and routine inspection"
            },
            "low": {
                "problem": "Early fan airflow performance deviation",
                "cause": "Mild change in rotational speed or torque may indicate airflow restriction or blade dust buildup",
                "solution": "Increase monitoring and inspect fan blades during the next planned maintenance"
            },
            "medium": {
                "problem": "Possible fan imbalance or bearing wear",
                "cause": "Abnormal speed or torque pattern may indicate blade imbalance, bearing wear, or airflow restriction",
                "solution": "Inspect fan blades, shaft alignment, bearing condition, and airflow path"
            },
            "high": {
                "problem": "Severe fan airflow imbalance or bearing stress",
                "cause": "Low health score may indicate high mechanical stress, bearing wear, or unstable rotation",
                "solution": "Reduce fan operation load if possible and schedule maintenance soon"
            },
            "critical": {
                "problem": "Critical fan operating condition",
                "cause": "Severe abnormal pattern may indicate bearing failure, blade damage, or unsafe vibration condition",
                "solution": "Stop fan if vibration is high and urgently inspect bearing, shaft alignment, and blade condition"
            }
        },

        "Turbine": {
            "normal": {
                "problem": "No significant turbine abnormality detected",
                "cause": "Turbine operating pattern appears stable",
                "solution": "Continue normal monitoring and trend recording"
            },
            "low": {
                "problem": "Early turbine performance variation",
                "cause": "Minor speed or temperature variation may indicate early load instability",
                "solution": "Monitor speed stability, lubrication, and temperature trend"
            },
            "medium": {
                "problem": "Possible turbine thermal or speed instability",
                "cause": "Abnormal speed or temperature pattern may indicate thermal stress or blade wear",
                "solution": "Inspect vibration trend, lubrication system, speed stability, and blade condition"
            },
            "high": {
                "problem": "High turbine mechanical stress",
                "cause": "Low health score may indicate speed instability, thermal stress, or high mechanical load",
                "solution": "Plan maintenance soon and verify safe operating limits"
            },
            "critical": {
                "problem": "Critical turbine condition",
                "cause": "Severe abnormal pattern may indicate unsafe speed instability, thermal stress, or blade damage",
                "solution": "Stop or isolate turbine if safety risk exists and perform urgent inspection"
            }
        },

        "Blower": {
            "normal": {
                "problem": "No significant blower abnormality detected",
                "cause": "Blower operating condition appears stable",
                "solution": "Continue routine monitoring"
            },
            "low": {
                "problem": "Early blower airflow deviation",
                "cause": "Small load or speed change may indicate filter restriction or airflow variation",
                "solution": "Check inlet filter and monitor outlet pressure"
            },
            "medium": {
                "problem": "Possible blower airflow restriction or bearing issue",
                "cause": "Abnormal torque or speed pattern may indicate blocked filter, bearing wear, or motor load increase",
                "solution": "Inspect inlet filter, outlet pressure, bearing condition, and motor load"
            },
            "high": {
                "problem": "High-risk blower operating condition",
                "cause": "Low health score may indicate airflow restriction, overheating, or bearing stress",
                "solution": "Plan maintenance soon and reduce blower load if possible"
            },
            "critical": {
                "problem": "Critical blower condition",
                "cause": "Severe abnormal pattern may indicate major airflow blockage, bearing failure, or overheating",
                "solution": "Stop blower if necessary and urgently inspect filter, bearings, and motor load"
            }
        },

        "Gearbox": {
            "normal": {
                "problem": "No significant gearbox abnormality detected",
                "cause": "Gearbox load and temperature pattern appear stable",
                "solution": "Continue normal operation and routine lubrication check"
            },
            "low": {
                "problem": "Early gearbox load variation",
                "cause": "Minor torque change may indicate early gear contact variation or lubrication change",
                "solution": "Monitor gearbox temperature and oil condition"
            },
            "medium": {
                "problem": "Possible gear wear or lubrication issue",
                "cause": "Abnormal torque or temperature pattern may indicate gear wear, misalignment, or insufficient lubrication",
                "solution": "Inspect oil level, gear teeth condition, vibration, and shaft alignment"
            },
            "high": {
                "problem": "High gearbox mechanical stress",
                "cause": "Low health score may indicate lubrication failure, gear wear, or abnormal torque transmission",
                "solution": "Schedule maintenance soon and avoid high load operation"
            },
            "critical": {
                "problem": "Critical gearbox condition",
                "cause": "Severe abnormal pattern may indicate gear damage, lubrication failure, or unsafe torque transmission",
                "solution": "Stop gearbox operation if necessary and perform urgent inspection"
            }
        },

        "Bearing Unit": {
            "normal": {
                "problem": "No significant bearing abnormality detected",
                "cause": "Bearing condition appears stable",
                "solution": "Continue normal monitoring"
            },
            "low": {
                "problem": "Early bearing friction variation",
                "cause": "Minor temperature or speed variation may indicate early lubrication change",
                "solution": "Monitor bearing temperature and lubrication condition"
            },
            "medium": {
                "problem": "Possible bearing wear or lubrication issue",
                "cause": "Abnormal speed, torque, or temperature may indicate bearing friction increase",
                "solution": "Inspect lubrication, bearing temperature, vibration, and shaft movement"
            },
            "high": {
                "problem": "High bearing wear risk",
                "cause": "Low health score may indicate friction increase, shaft imbalance, or lubrication failure",
                "solution": "Plan bearing inspection and prepare replacement if wear is confirmed"
            },
            "critical": {
                "problem": "Critical bearing condition",
                "cause": "Severe abnormal pattern may indicate bearing failure or unsafe shaft movement",
                "solution": "Stop equipment if vibration or temperature is high and replace bearing if required"
            }
        },

        "Conveyor": {
            "normal": {
                "problem": "No significant conveyor abnormality detected",
                "cause": "Conveyor drive and load pattern appear stable",
                "solution": "Continue routine monitoring"
            },
            "low": {
                "problem": "Early conveyor load variation",
                "cause": "Minor torque or speed variation may indicate belt tension change or material load variation",
                "solution": "Monitor belt movement and check for small misalignment"
            },
            "medium": {
                "problem": "Possible belt misalignment or roller wear",
                "cause": "Abnormal torque or speed pattern may indicate belt tension issue, roller wear, or material blockage",
                "solution": "Inspect belt tension, roller condition, drive motor load, and material flow"
            },
            "high": {
                "problem": "High conveyor drive stress",
                "cause": "Low health score may indicate belt misalignment, roller jam, or overload",
                "solution": "Reduce conveyor load and schedule maintenance soon"
            },
            "critical": {
                "problem": "Critical conveyor condition",
                "cause": "Severe abnormal pattern may indicate belt jam, roller failure, or unsafe drive motor stress",
                "solution": "Stop conveyor and inspect belt, rollers, drive motor, and blockage immediately"
            }
        },

        "Hydraulic System": {
            "normal": {
                "problem": "No significant hydraulic abnormality detected",
                "cause": "Hydraulic operating pattern appears stable",
                "solution": "Continue routine monitoring"
            },
            "low": {
                "problem": "Early hydraulic pressure variation",
                "cause": "Minor load or temperature change may indicate early fluid condition change",
                "solution": "Monitor hydraulic pressure and oil temperature"
            },
            "medium": {
                "problem": "Possible hydraulic pressure instability",
                "cause": "Abnormal torque or temperature pattern may indicate fluid leakage, filter restriction, or pump overload",
                "solution": "Inspect pressure, oil level, hoses, seals, filters, and actuator movement"
            },
            "high": {
                "problem": "High hydraulic system risk",
                "cause": "Low health score may indicate pressure instability, overheating, or leakage",
                "solution": "Plan maintenance soon and reduce hydraulic load if possible"
            },
            "critical": {
                "problem": "Critical hydraulic system condition",
                "cause": "Severe abnormal pattern may indicate major leakage, pump failure, or unsafe pressure condition",
                "solution": "Stop hydraulic operation if unsafe and urgently inspect hoses, seals, pressure, and pump"
            }
        },

        "Heat Exchanger": {
            "normal": {
                "problem": "No significant heat exchanger abnormality detected",
                "cause": "Temperature pattern appears stable",
                "solution": "Continue routine monitoring"
            },
            "low": {
                "problem": "Early heat transfer efficiency change",
                "cause": "Minor temperature variation may indicate early fouling or flow change",
                "solution": "Monitor inlet and outlet temperature difference"
            },
            "medium": {
                "problem": "Possible fouling or flow restriction",
                "cause": "Abnormal temperature pattern may indicate reduced heat transfer efficiency",
                "solution": "Inspect temperature difference, flow restriction, and cleaning requirement"
            },
            "high": {
                "problem": "High heat exchanger efficiency loss",
                "cause": "Low health score may indicate fouling, blocked flow, or abnormal temperature difference",
                "solution": "Schedule cleaning and inspect inlet/outlet flow condition soon"
            },
            "critical": {
                "problem": "Critical heat exchanger condition",
                "cause": "Severe abnormal pattern may indicate major blockage or thermal performance failure",
                "solution": "Stop or bypass if required and perform urgent cleaning and flow inspection"
            }
        },

        "Industrial Valve": {
            "normal": {
                "problem": "No significant valve abnormality detected",
                "cause": "Valve operating pattern appears stable",
                "solution": "Continue routine monitoring"
            },
            "low": {
                "problem": "Early valve response variation",
                "cause": "Minor load or temperature change may indicate early actuator delay or flow variation",
                "solution": "Monitor valve response and check actuator movement during inspection"
            },
            "medium": {
                "problem": "Possible valve sticking or leakage",
                "cause": "Abnormal operating pattern may indicate actuator issue, seal wear, or flow restriction",
                "solution": "Inspect valve position, actuator response, leakage, pressure drop, and seal condition"
            },
            "high": {
                "problem": "High valve operation risk",
                "cause": "Low health score may indicate valve sticking, leakage, or actuator stress",
                "solution": "Plan maintenance soon and verify valve operation safely"
            },
            "critical": {
                "problem": "Critical valve condition",
                "cause": "Severe abnormal pattern may indicate stuck valve, severe leakage, or actuator failure",
                "solution": "Isolate valve if required and urgently inspect actuator, seal, and pressure drop"
            }
        }
    }

    return context.get(equipment_category, context["Motor"])


def get_health_band(health_score):
    """
    Convert health score into scenario band.
    """

    if health_score >= 80:
        return "normal"
    elif health_score >= 60:
        return "low"
    elif health_score >= 40:
        return "medium"
    elif health_score >= 20:
        return "high"
    else:
        return "critical"


def detect_abnormal_sensors(row):
    """
    Detect abnormal sensor patterns from scaled values.

    The AI4I dataset provides the same numeric features for every simulated
    equipment type. To avoid illogical wording, this function converts those
    abnormal values into equipment-specific maintenance language.

    Example:
    - Heat Exchanger focuses on temperature / thermal performance.
    - Pump focuses on hydraulic load, speed, and temperature.
    - Fan / Blower focuses on airflow, rotation, load, and temperature.
    """

    equipment = str(row.get("equipment_category", "")).lower()
    abnormal_sensors = []

    def sensor_value(column_name):
        try:
            return float(row.get(column_name, 0))
        except Exception:
            return 0.0

    def add_if_abnormal(value, very_high, high, low, very_low=None):
        if value > 1.5:
            abnormal_sensors.append(very_high)
        elif value > 1:
            abnormal_sensors.append(high)
        elif value < -1.5:
            abnormal_sensors.append(very_low if very_low else low)
        elif value < -1:
            abnormal_sensors.append(low)

    torque = sensor_value("Torque [Nm]")
    tool_wear = sensor_value("Tool wear [min]")
    speed = sensor_value("Rotational speed [rpm]")
    air_temp = sensor_value("Air temperature [K]")
    process_temp = sensor_value("Process temperature [K]")

    # Heat exchangers are not rotating assets, so avoid torque/speed wording.
    if "heat" in equipment or "exchanger" in equipment:
        add_if_abnormal(
            air_temp,
            "very high air temperature",
            "high air temperature",
            "low air temperature",
            "very low air temperature"
        )
        add_if_abnormal(
            process_temp,
            "very high process temperature",
            "high process temperature",
            "low process temperature",
            "very low process temperature"
        )
        return abnormal_sensors

    # Industrial valves are better described using actuator/load and thermal wording.
    if "valve" in equipment:
        add_if_abnormal(
            torque,
            "very high actuator load demand",
            "high actuator load demand",
            "low actuator load response",
            "very low actuator load response"
        )
        add_if_abnormal(
            air_temp,
            "very high operating temperature",
            "high operating temperature",
            "low operating temperature",
            "very low operating temperature"
        )
        add_if_abnormal(
            process_temp,
            "very high process temperature",
            "high process temperature",
            "low process temperature",
            "very low process temperature"
        )
        return abnormal_sensors

    # Hydraulic systems are described using load/pressure-style language.
    if "hydraulic" in equipment:
        add_if_abnormal(
            torque,
            "very high hydraulic load demand",
            "high hydraulic load demand",
            "low hydraulic load response",
            "very low hydraulic load response"
        )
        add_if_abnormal(
            air_temp,
            "very high operating temperature",
            "high operating temperature",
            "low operating temperature",
            "very low operating temperature"
        )
        add_if_abnormal(
            process_temp,
            "very high process temperature",
            "high process temperature",
            "low process temperature",
            "very low process temperature"
        )
        return abnormal_sensors

    # Fan and blower wording should focus on airflow, rotation, load, and temperature.
    if "fan" in equipment or "blower" in equipment:
        add_if_abnormal(
            speed,
            "very high rotational airflow speed",
            "high rotational airflow speed",
            "low rotational airflow speed",
            "very low rotational airflow speed"
        )
        add_if_abnormal(
            torque,
            "very high airflow load demand",
            "high airflow load demand",
            "low airflow load demand",
            "very low airflow load demand"
        )
        add_if_abnormal(
            air_temp,
            "very high air temperature",
            "high air temperature",
            "low air temperature",
            "very low air temperature"
        )
        add_if_abnormal(
            process_temp,
            "very high process temperature",
            "high process temperature",
            "low process temperature",
            "very low process temperature"
        )
        return abnormal_sensors

    # Gearbox and bearing units are mainly mechanical wear/load assets.
    if "gear" in equipment or "bearing" in equipment:
        add_if_abnormal(
            torque,
            "very high mechanical load",
            "high mechanical load",
            "low mechanical load",
            "very low mechanical load"
        )
        add_if_abnormal(
            speed,
            "very high shaft speed",
            "high shaft speed",
            "low shaft speed",
            "very low shaft speed"
        )
        if tool_wear > 1.5:
            abnormal_sensors.append("very high wear indicator")
        elif tool_wear > 1:
            abnormal_sensors.append("high wear indicator")
        add_if_abnormal(
            process_temp,
            "very high operating temperature",
            "high operating temperature",
            "low operating temperature",
            "very low operating temperature"
        )
        return abnormal_sensors

    # Conveyor wording should avoid generic machine language and focus on drive/load.
    if "conveyor" in equipment:
        add_if_abnormal(
            torque,
            "very high conveyor drive load",
            "high conveyor drive load",
            "low conveyor drive load",
            "very low conveyor drive load"
        )
        add_if_abnormal(
            speed,
            "very high conveyor speed variation",
            "high conveyor speed variation",
            "low conveyor speed variation",
            "very low conveyor speed variation"
        )
        if tool_wear > 1.5:
            abnormal_sensors.append("very high wear indicator")
        elif tool_wear > 1:
            abnormal_sensors.append("high wear indicator")
        return abnormal_sensors

    # Default rotating equipment: Motor, Pump, Compressor, Turbine, and fallback.
    add_if_abnormal(
        torque,
        "very high torque",
        "high torque",
        "low torque",
        "very low torque"
    )

    if tool_wear > 1.5:
        abnormal_sensors.append("very high tool wear")
    elif tool_wear > 1:
        abnormal_sensors.append("high tool wear")

    add_if_abnormal(
        speed,
        "very high rotational speed",
        "high rotational speed",
        "low rotational speed",
        "very low rotational speed"
    )

    add_if_abnormal(
        air_temp,
        "very high air temperature",
        "high air temperature",
        "low air temperature",
        "very low air temperature"
    )

    add_if_abnormal(
        process_temp,
        "very high process temperature",
        "high process temperature",
        "low process temperature",
        "very low process temperature"
    )

    return abnormal_sensors

def generate_problem_cause_solution(row):
    """
    Generate problem, cause, and solution based on:
    - equipment type
    - health score band
    - abnormal sensor pattern
    """

    equipment_category = row["equipment_category"]
    health_score = row["health_score"]
    health_band = get_health_band(health_score)

    context = get_equipment_context(equipment_category)
    scenario = context[health_band]

    problem_detected = scenario["problem"]
    probable_cause = scenario["cause"]
    recommended_solution = scenario["solution"]

    return problem_detected, probable_cause, recommended_solution


def generate_short_reason(row):
    """
    Generate short abnormal pattern reason for table and report.
    """

    abnormal_sensors = detect_abnormal_sensors(row)
    reasons = []

    if row["if_anomaly_label"] == 1:
        reasons.append("anomaly detected")

    if abnormal_sensors:
        reasons.extend(abnormal_sensors[:3])

    if not reasons:
        return "health score based condition"

    return ", ".join(reasons)


def generate_recommended_action(row):
    """
    Keep recommended action short and clean.
    It should not repeat problem/cause details.
    """

    return row["recommended_solution"]


def generate_explanation(row):
    """
    Generate short AI explanation without repeating problem/cause/solution.
    """

    alert_level = row["alert_level"]

    if row["if_anomaly_label"] == 1:
        model_part = (
            "The AI system compared this machine record with the normal operating pattern "
            "and detected abnormal sensor behavior."
        )
    else:
        model_part = (
            "The AI system compared this machine record with the normal operating pattern. "
            "No strong anomaly was detected, but the health score is still used for condition monitoring."
        )

    return (
        f"{model_part} Based on the health score and detected sensor behavior, "
        f"the machine was classified as {alert_level}. "
        "This result should support maintenance decision-making, but the final action should still be confirmed by an engineer or technician."
    )


def apply_explainability(machine_results):
    """
    Add clean problem, cause, solution, short reason, action, and AI explanation.
    """

    df = machine_results.copy()

    generated_values = df.apply(generate_problem_cause_solution, axis=1)

    df["problem_detected"] = generated_values.apply(lambda x: x[0])
    df["probable_cause"] = generated_values.apply(lambda x: x[1])
    df["recommended_solution"] = generated_values.apply(lambda x: x[2])

    df["short_reason"] = df.apply(generate_short_reason, axis=1)
    df["recommended_action"] = df.apply(generate_recommended_action, axis=1)
    df["explanation"] = df.apply(generate_explanation, axis=1)

    print("\nExplainability output generated successfully.")
    print("Clean problem, cause, solution, action, and explanation columns added.")

    return df