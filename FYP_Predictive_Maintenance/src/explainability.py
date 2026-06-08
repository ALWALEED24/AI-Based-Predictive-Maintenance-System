def get_equipment_context(equipment_category):
    """
    Equipment-specific maintenance knowledge.

    This file is the rule-based decision-support layer that runs after the
    machine learning output. The ML model detects abnormal behaviour, while
    this layer explains the likely maintenance meaning in simple engineering
    language.
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

        "Generator": {
            "normal": {
                "problem": "No significant generator abnormality detected",
                "cause": "Generator load, speed, and temperature pattern appear stable",
                "solution": "Continue normal operation and routine electrical monitoring"
            },
            "low": {
                "problem": "Early generator load variation",
                "cause": "Minor change in rotational speed, torque, or temperature may indicate early electrical load fluctuation",
                "solution": "Monitor generator load, output stability, temperature, and cooling condition"
            },
            "medium": {
                "problem": "Possible generator overload or cooling issue",
                "cause": "Abnormal torque, speed, or temperature pattern may indicate load imbalance, cooling limitation, or bearing stress",
                "solution": "Inspect load balance, cooling airflow, bearing condition, and electrical output stability"
            },
            "high": {
                "problem": "High generator operating stress",
                "cause": "Low health score may indicate overload, overheating, bearing stress, or unstable electrical output",
                "solution": "Reduce generator load if possible and schedule maintenance soon"
            },
            "critical": {
                "problem": "Critical generator condition",
                "cause": "Severe abnormal pattern may indicate overheating, overload, bearing failure, or unsafe electrical operation",
                "solution": "Isolate generator if safety risk exists and perform urgent electrical and mechanical inspection"
            }
        },

        "Chiller": {
            "normal": {
                "problem": "No significant chiller abnormality detected",
                "cause": "Chiller temperature and operating load pattern appear stable",
                "solution": "Continue routine monitoring and normal cooling operation"
            },
            "low": {
                "problem": "Early chiller efficiency variation",
                "cause": "Minor temperature or load change may indicate early cooling efficiency reduction",
                "solution": "Monitor chilled water temperature, condenser condition, and operating load"
            },
            "medium": {
                "problem": "Possible chiller cooling inefficiency",
                "cause": "Abnormal temperature or load pattern may indicate condenser fouling, refrigerant issue, or compressor load stress",
                "solution": "Inspect condenser cleanliness, refrigerant condition, cooling water flow, and compressor operation"
            },
            "high": {
                "problem": "High chiller performance risk",
                "cause": "Low health score may indicate cooling efficiency loss, overheating, or unstable compressor load",
                "solution": "Schedule maintenance soon and check condenser, refrigerant level, and cooling flow"
            },
            "critical": {
                "problem": "Critical chiller condition",
                "cause": "Severe abnormal pattern may indicate major cooling failure, compressor stress, or thermal control problem",
                "solution": "Reduce cooling load or isolate chiller if required and perform urgent inspection"
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

        "Cooling Tower": {
            "normal": {
                "problem": "No significant cooling tower abnormality detected",
                "cause": "Cooling tower temperature and airflow pattern appear stable",
                "solution": "Continue routine monitoring and cooling water checks"
            },
            "low": {
                "problem": "Early cooling tower performance variation",
                "cause": "Minor airflow or temperature change may indicate early cooling efficiency reduction",
                "solution": "Monitor cooling water temperature, fan operation, and water distribution"
            },
            "medium": {
                "problem": "Possible cooling tower efficiency loss",
                "cause": "Abnormal temperature or airflow pattern may indicate fan issue, scaling, fouling, or poor water distribution",
                "solution": "Inspect fan operation, fill media, spray nozzles, basin condition, and water quality"
            },
            "high": {
                "problem": "High cooling tower performance risk",
                "cause": "Low health score may indicate cooling efficiency loss, airflow restriction, or scaling buildup",
                "solution": "Schedule cleaning and inspect fan, fill media, and water distribution system"
            },
            "critical": {
                "problem": "Critical cooling tower condition",
                "cause": "Severe abnormal pattern may indicate major cooling failure, fan failure, or serious flow restriction",
                "solution": "Reduce cooling load if required and perform urgent cooling tower inspection"
            }
        },

        "Centrifuge": {
            "normal": {
                "problem": "No significant centrifuge abnormality detected",
                "cause": "Centrifuge speed, load, and temperature pattern appear stable",
                "solution": "Continue normal operation and routine vibration monitoring"
            },
            "low": {
                "problem": "Early centrifuge balance variation",
                "cause": "Minor speed or load change may indicate early imbalance or feed variation",
                "solution": "Monitor speed stability, vibration trend, and feed consistency"
            },
            "medium": {
                "problem": "Possible centrifuge imbalance or bearing wear",
                "cause": "Abnormal speed or torque pattern may indicate rotor imbalance, bearing wear, or unstable feed load",
                "solution": "Inspect rotor balance, bearing condition, vibration, and feed distribution"
            },
            "high": {
                "problem": "High centrifuge mechanical stress",
                "cause": "Low health score may indicate rotor imbalance, high vibration risk, or bearing stress",
                "solution": "Reduce load if possible and schedule centrifuge mechanical inspection soon"
            },
            "critical": {
                "problem": "Critical centrifuge condition",
                "cause": "Severe abnormal pattern may indicate unsafe rotor imbalance, bearing failure, or high vibration condition",
                "solution": "Stop centrifuge if unsafe and perform urgent rotor, bearing, and vibration inspection"
            }
        },

        "Mixer": {
            "normal": {
                "problem": "No significant mixer abnormality detected",
                "cause": "Mixer load and speed pattern appear stable",
                "solution": "Continue normal operation and routine mechanical checks"
            },
            "low": {
                "problem": "Early mixer load variation",
                "cause": "Minor torque or speed variation may indicate viscosity change or early shaft load variation",
                "solution": "Monitor mixing load, product viscosity, and shaft movement"
            },
            "medium": {
                "problem": "Possible mixer overload or shaft misalignment",
                "cause": "Abnormal torque or speed pattern may indicate high viscosity, blade resistance, or shaft alignment issue",
                "solution": "Inspect mixing blades, shaft alignment, coupling, and process load"
            },
            "high": {
                "problem": "High mixer drive stress",
                "cause": "Low health score may indicate overload, blade resistance, or abnormal torque demand",
                "solution": "Reduce mixer load if possible and schedule mechanical inspection soon"
            },
            "critical": {
                "problem": "Critical mixer condition",
                "cause": "Severe abnormal pattern may indicate jammed impeller, shaft failure, or unsafe motor load",
                "solution": "Stop mixer if necessary and urgently inspect blades, shaft, coupling, and drive motor"
            }
        },

        "Agitator": {
            "normal": {
                "problem": "No significant agitator abnormality detected",
                "cause": "Agitator drive and mixing pattern appear stable",
                "solution": "Continue routine monitoring"
            },
            "low": {
                "problem": "Early agitator load variation",
                "cause": "Minor torque or speed change may indicate process load or viscosity variation",
                "solution": "Monitor agitator load, shaft condition, and product consistency"
            },
            "medium": {
                "problem": "Possible agitator shaft or blade stress",
                "cause": "Abnormal torque or speed pattern may indicate blade fouling, shaft misalignment, or high process resistance",
                "solution": "Inspect agitator blades, shaft alignment, coupling, and tank process condition"
            },
            "high": {
                "problem": "High agitator mechanical stress",
                "cause": "Low health score may indicate blade resistance, overload, or shaft stress",
                "solution": "Schedule inspection soon and reduce agitator load if possible"
            },
            "critical": {
                "problem": "Critical agitator condition",
                "cause": "Severe abnormal pattern may indicate shaft damage, blade jam, or unsafe drive stress",
                "solution": "Stop agitator if unsafe and urgently inspect shaft, blades, coupling, and motor"
            }
        },

        "Vacuum Pump": {
            "normal": {
                "problem": "No significant vacuum pump abnormality detected",
                "cause": "Vacuum pump load and temperature pattern appear stable",
                "solution": "Continue normal operation and vacuum level monitoring"
            },
            "low": {
                "problem": "Early vacuum pump performance deviation",
                "cause": "Minor load or temperature change may indicate early vacuum efficiency variation",
                "solution": "Monitor vacuum level, seal condition, and pump temperature"
            },
            "medium": {
                "problem": "Possible vacuum loss or pump wear",
                "cause": "Abnormal torque or temperature pattern may indicate seal leakage, filter restriction, or internal wear",
                "solution": "Inspect vacuum level, seals, filters, oil condition, and pump wear"
            },
            "high": {
                "problem": "High vacuum pump operating risk",
                "cause": "Low health score may indicate vacuum loss, overheating, internal wear, or seal leakage",
                "solution": "Schedule maintenance soon and check vacuum performance and sealing condition"
            },
            "critical": {
                "problem": "Critical vacuum pump condition",
                "cause": "Severe abnormal pattern may indicate major vacuum loss, overheating, or internal pump failure",
                "solution": "Stop or isolate vacuum pump if required and perform urgent inspection"
            }
        },

        "Separator": {
            "normal": {
                "problem": "No significant separator abnormality detected",
                "cause": "Separator operating load and thermal pattern appear stable",
                "solution": "Continue normal monitoring and routine process checks"
            },
            "low": {
                "problem": "Early separator process variation",
                "cause": "Minor temperature or load change may indicate early flow or process condition variation",
                "solution": "Monitor inlet flow, temperature, pressure trend, and separation performance"
            },
            "medium": {
                "problem": "Possible separator flow instability",
                "cause": "Abnormal load or temperature pattern may indicate inlet flow variation, internal restriction, or process imbalance",
                "solution": "Inspect inlet flow, outlet condition, internal elements, pressure trend, and process balance"
            },
            "high": {
                "problem": "High separator process risk",
                "cause": "Low health score may indicate unstable flow, internal restriction, or abnormal process load",
                "solution": "Schedule inspection soon and verify separator pressure, flow, and outlet quality"
            },
            "critical": {
                "problem": "Critical separator condition",
                "cause": "Severe abnormal pattern may indicate major process instability, blockage, or unsafe operating condition",
                "solution": "Isolate separator if safety risk exists and perform urgent process and mechanical inspection"
            }
        }
    }

    # Robust matching: protects against extra spaces, different casing, or
    # category names such as "Industrial Motor" / "Cooling tower fan".
    equipment_text = str(equipment_category).strip().lower()

    for category_name in context.keys():
        if category_name.lower() == equipment_text:
            return context[category_name]

    alias_rules = [
        ("cooling tower", "Cooling Tower"),
        ("vacuum pump", "Vacuum Pump"),
        ("heat exchanger", "Heat Exchanger"),
        ("hydraulic", "Hydraulic System"),
        ("compressor", "Compressor"),
        ("conveyor", "Conveyor"),
        ("centrifuge", "Centrifuge"),
        ("agitator", "Agitator"),
        ("separator", "Separator"),
        ("generator", "Generator"),
        ("chiller", "Chiller"),
        ("gear", "Gearbox"),
        ("pump", "Pump"),
        ("motor", "Motor"),
        ("fan", "Fan"),
        ("blower", "Blower"),
        ("turbine", "Turbine"),
        ("mixer", "Mixer"),
    ]

    for keyword, category_name in alias_rules:
        if keyword in equipment_text:
            return context[category_name]

    return context["Motor"]


def get_health_band(health_score):
    """
    Convert health score into scenario band.
    """

    try:
        health_score = float(health_score)
    except Exception:
        health_score = 100.0

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


def get_numeric_value(row, column_name, default=0.0):
    """
    Safely read a numeric value from a pandas row.
    """

    try:
        value = row.get(column_name, default)
        if value is None:
            return default
        return float(value)
    except Exception:
        return default


def get_scaled_sensor_value(row, original_column_name):
    """
    Read the scaled version of a sensor if available.

    After the technical foundation fix, the project keeps:
    - original readable values for dashboard/report display
    - scaled_* values for AI/model/explainability logic

    This function prevents the rule-based explanation from accidentally treating
    real values such as 298 K or 1500 rpm as abnormal z-score values.
    """

    scaled_column = f"scaled_{original_column_name}"

    if scaled_column in row.index:
        return get_numeric_value(row, scaled_column, default=0.0)

    # Backward compatibility for older outputs where the displayed columns were
    # still scaled values.
    return get_numeric_value(row, original_column_name, default=0.0)


def get_anomaly_label(row):
    """
    Safely read the final anomaly flag.
    """

    if "if_anomaly_label" in row.index:
        return int(get_numeric_value(row, "if_anomaly_label", default=0))

    if "ai_anomaly_vote_count" in row.index:
        return 1 if get_numeric_value(row, "ai_anomaly_vote_count", default=0) > 0 else 0

    return 0


def detect_abnormal_sensors(row):
    """
    Detect abnormal sensor patterns from scaled values.

    Important:
    - Use scaled_* columns for anomaly explanation.
    - Use original columns only as a fallback for older pipeline outputs.
    - This keeps the dashboard/report readable while keeping the rule logic correct.
    """

    equipment = str(row.get("equipment_category", "")).lower()
    abnormal_sensors = []

    def add_if_abnormal(value, very_high, high, low, very_low=None):
        if value > 1.5:
            abnormal_sensors.append(very_high)
        elif value > 1:
            abnormal_sensors.append(high)
        elif value < -1.5:
            abnormal_sensors.append(very_low if very_low else low)
        elif value < -1:
            abnormal_sensors.append(low)

    torque = get_scaled_sensor_value(row, "Torque [Nm]")
    tool_wear = get_scaled_sensor_value(row, "Tool wear [min]")
    speed = get_scaled_sensor_value(row, "Rotational speed [rpm]")
    air_temp = get_scaled_sensor_value(row, "Air temperature [K]")
    process_temp = get_scaled_sensor_value(row, "Process temperature [K]")

    if "heat" in equipment or "exchanger" in equipment:
        add_if_abnormal(air_temp, "very high air temperature", "high air temperature", "low air temperature", "very low air temperature")
        add_if_abnormal(process_temp, "very high process temperature", "high process temperature", "low process temperature", "very low process temperature")
        return abnormal_sensors

    if "cooling tower" in equipment:
        add_if_abnormal(air_temp, "very high cooling tower air temperature", "high cooling tower air temperature", "low cooling tower air temperature", "very low cooling tower air temperature")
        add_if_abnormal(process_temp, "very high cooling water temperature", "high cooling water temperature", "low cooling water temperature", "very low cooling water temperature")
        add_if_abnormal(speed, "very high fan speed variation", "high fan speed variation", "low fan speed variation", "very low fan speed variation")
        return abnormal_sensors

    if "chiller" in equipment:
        add_if_abnormal(air_temp, "very high cooling-side temperature", "high cooling-side temperature", "low cooling-side temperature", "very low cooling-side temperature")
        add_if_abnormal(process_temp, "very high process cooling temperature", "high process cooling temperature", "low process cooling temperature", "very low process cooling temperature")
        add_if_abnormal(torque, "very high compressor load demand", "high compressor load demand", "low compressor load demand", "very low compressor load demand")
        return abnormal_sensors

    if "hydraulic" in equipment:
        add_if_abnormal(torque, "very high hydraulic load demand", "high hydraulic load demand", "low hydraulic load response", "very low hydraulic load response")
        add_if_abnormal(air_temp, "very high operating temperature", "high operating temperature", "low operating temperature", "very low operating temperature")
        add_if_abnormal(process_temp, "very high process temperature", "high process temperature", "low process temperature", "very low process temperature")
        return abnormal_sensors

    if "fan" in equipment or "blower" in equipment:
        add_if_abnormal(speed, "very high rotational airflow speed", "high rotational airflow speed", "low rotational airflow speed", "very low rotational airflow speed")
        add_if_abnormal(torque, "very high airflow load demand", "high airflow load demand", "low airflow load demand", "very low airflow load demand")
        add_if_abnormal(air_temp, "very high air temperature", "high air temperature", "low air temperature", "very low air temperature")
        add_if_abnormal(process_temp, "very high process temperature", "high process temperature", "low process temperature", "very low process temperature")
        return abnormal_sensors

    if "gear" in equipment:
        add_if_abnormal(torque, "very high mechanical load", "high mechanical load", "low mechanical load", "very low mechanical load")
        add_if_abnormal(speed, "very high shaft speed", "high shaft speed", "low shaft speed", "very low shaft speed")
        if tool_wear > 1.5:
            abnormal_sensors.append("very high wear indicator")
        elif tool_wear > 1:
            abnormal_sensors.append("high wear indicator")
        add_if_abnormal(process_temp, "very high operating temperature", "high operating temperature", "low operating temperature", "very low operating temperature")
        return abnormal_sensors

    if "conveyor" in equipment:
        add_if_abnormal(torque, "very high conveyor drive load", "high conveyor drive load", "low conveyor drive load", "very low conveyor drive load")
        add_if_abnormal(speed, "very high conveyor speed variation", "high conveyor speed variation", "low conveyor speed variation", "very low conveyor speed variation")
        if tool_wear > 1.5:
            abnormal_sensors.append("very high wear indicator")
        elif tool_wear > 1:
            abnormal_sensors.append("high wear indicator")
        return abnormal_sensors

    if "generator" in equipment:
        add_if_abnormal(torque, "very high generator load demand", "high generator load demand", "low generator load response", "very low generator load response")
        add_if_abnormal(speed, "very high rotational speed instability", "high rotational speed instability", "low rotational speed instability", "very low rotational speed instability")
        add_if_abnormal(air_temp, "very high generator cooling temperature", "high generator cooling temperature", "low generator cooling temperature", "very low generator cooling temperature")
        add_if_abnormal(process_temp, "very high operating temperature", "high operating temperature", "low operating temperature", "very low operating temperature")
        return abnormal_sensors

    if "centrifuge" in equipment:
        add_if_abnormal(speed, "very high centrifuge speed instability", "high centrifuge speed instability", "low centrifuge speed instability", "very low centrifuge speed instability")
        add_if_abnormal(torque, "very high centrifuge load demand", "high centrifuge load demand", "low centrifuge load demand", "very low centrifuge load demand")
        add_if_abnormal(process_temp, "very high operating temperature", "high operating temperature", "low operating temperature", "very low operating temperature")
        return abnormal_sensors

    if "mixer" in equipment or "agitator" in equipment:
        add_if_abnormal(torque, "very high mixing load demand", "high mixing load demand", "low mixing load demand", "very low mixing load demand")
        add_if_abnormal(speed, "very high shaft speed variation", "high shaft speed variation", "low shaft speed variation", "very low shaft speed variation")
        if tool_wear > 1.5:
            abnormal_sensors.append("very high mechanical wear indicator")
        elif tool_wear > 1:
            abnormal_sensors.append("high mechanical wear indicator")
        return abnormal_sensors

    if "vacuum pump" in equipment:
        add_if_abnormal(torque, "very high vacuum pump load demand", "high vacuum pump load demand", "low vacuum pump load response", "very low vacuum pump load response")
        add_if_abnormal(air_temp, "very high pump operating temperature", "high pump operating temperature", "low pump operating temperature", "very low pump operating temperature")
        add_if_abnormal(process_temp, "very high process temperature", "high process temperature", "low process temperature", "very low process temperature")
        return abnormal_sensors

    if "separator" in equipment:
        add_if_abnormal(torque, "very high separator process load", "high separator process load", "low separator process load", "very low separator process load")
        add_if_abnormal(process_temp, "very high separator process temperature", "high separator process temperature", "low separator process temperature", "very low separator process temperature")
        add_if_abnormal(air_temp, "very high operating temperature", "high operating temperature", "low operating temperature", "very low operating temperature")
        return abnormal_sensors

    add_if_abnormal(torque, "very high torque", "high torque", "low torque", "very low torque")

    if tool_wear > 1.5:
        abnormal_sensors.append("very high tool wear")
    elif tool_wear > 1:
        abnormal_sensors.append("high tool wear")

    add_if_abnormal(speed, "very high rotational speed", "high rotational speed", "low rotational speed", "very low rotational speed")
    add_if_abnormal(air_temp, "very high air temperature", "high air temperature", "low air temperature", "very low air temperature")
    add_if_abnormal(process_temp, "very high process temperature", "high process temperature", "low process temperature", "very low process temperature")

    return abnormal_sensors


def generate_problem_cause_solution(row):
    """
    Generate problem, cause, and solution based on:
    - equipment type
    - health score band
    - rule-based maintenance knowledge
    """

    equipment_category = row.get("equipment_category", "Motor")
    health_score = row.get("health_score", 100)
    health_band = get_health_band(health_score)

    context = get_equipment_context(equipment_category)
    scenario = context[health_band]

    problem_detected = scenario["problem"]
    probable_cause = scenario["cause"]
    recommended_solution = scenario["solution"]

    return problem_detected, probable_cause, recommended_solution


def generate_short_reason(row):
    """
    Generate a short abnormal pattern reason for dashboard table and report.
    """

    health_band = get_health_band(row.get("health_score", 100))
    abnormal_sensors = detect_abnormal_sensors(row)
    reasons = []

    if get_anomaly_label(row) == 1:
        reasons.append("anomaly detected")

    if abnormal_sensors:
        reasons.extend(abnormal_sensors[:3])

    if reasons:
        return ", ".join(reasons)

    if health_band == "normal":
        return "No significant abnormality detected"

    return "Health score based condition"


def generate_recommended_action(row):
    """
    Keep recommended action short and clean.
    It should not repeat problem/cause details.
    """

    return row.get("recommended_solution", "Continue monitoring and schedule inspection if abnormal behaviour continues")


def generate_explanation(row):
    """
    Generate a short rule-based AI monitoring explanation without repeating
    the same problem/cause/solution paragraphs.
    """

    alert_level = row.get("alert_level", "Normal")
    health_score = row.get("health_score", "N/A")

    abnormal_sensors = detect_abnormal_sensors(row)
    abnormal_text = ", ".join(abnormal_sensors[:3]) if abnormal_sensors else "no major abnormal sensor pattern"

    if get_anomaly_label(row) == 1:
        model_part = (
            "The AI monitoring system compared this machine record with learned operational patterns "
            "and detected abnormal sensor behaviour."
        )
    else:
        model_part = (
            "The AI monitoring system compared this machine record with learned operational patterns. "
            "No strong anomaly was detected, but the health score is still used as an operational risk indicator."
        )

    return (
        f"{model_part} The machine health score is {health_score}, and the observed pattern is: "
        f"{abnormal_text}. Based on the health score, anomaly result, and equipment-specific rule layer, "
        f"the machine was classified as {alert_level}. This result supports condition-based maintenance "
        "decision-making, but the final maintenance action should still be confirmed by an engineer or technician."
    )


def apply_explainability(machine_results):
    """
    Add clean problem, cause, solution, short reason, action, and AI explanation.

    Output columns added:
    - problem_detected
    - probable_cause
    - recommended_solution
    - short_reason
    - recommended_action
    - explanation
    """

    df = machine_results.copy()

    if "equipment_category" not in df.columns:
        df["equipment_category"] = "Motor"

    if "health_score" not in df.columns:
        df["health_score"] = 100

    if "alert_level" not in df.columns:
        df["alert_level"] = "Normal"

    if "if_anomaly_label" not in df.columns:
        df["if_anomaly_label"] = 0

    generated_values = df.apply(generate_problem_cause_solution, axis=1)

    df["problem_detected"] = generated_values.apply(lambda x: x[0])
    df["probable_cause"] = generated_values.apply(lambda x: x[1])
    df["recommended_solution"] = generated_values.apply(lambda x: x[2])

    df["short_reason"] = df.apply(generate_short_reason, axis=1)
    df["recommended_action"] = df.apply(generate_recommended_action, axis=1)
    df["explanation"] = df.apply(generate_explanation, axis=1)

    print("\nExplainability output generated successfully.")
    print("Problem, cause, solution, action, and explanation columns added.")
    print("Rule-based explanation uses scaled_* sensor columns when available.")

    return df
