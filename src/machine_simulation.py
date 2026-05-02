import os


def create_simulated_machines():
    """
    Create simulated machine list for dashboard and reporting.

    Total machines: 60
    Categories: 12 equipment categories x 5 machines each

    These machine categories are used for prototype simulation only.
    They are not original labels from the AI4I 2020 dataset.
    """

    machine_categories = {
        "Motor": "MOT",
        "Pump": "PMP",
        "Compressor": "CMP",
        "Fan": "FAN",
        "Turbine": "TRB",
        "Blower": "BLW",
        "Gearbox": "GRB",
        "Bearing Unit": "BRG",
        "Conveyor": "CNV",
        "Hydraulic System": "HYD",
        "Heat Exchanger": "HEX",
        "Industrial Valve": "VLV"
    }

    machines = []

    for category, prefix in machine_categories.items():
        for i in range(1, 6):
            machines.append({
                "machine_id": f"{prefix}-{i:03d}",
                "equipment_category": category
            })

    return machines


def assign_machines_to_records(final_machine_results):
    """
    Assign simulated machine IDs and equipment categories to dataset records.
    The assignment repeats across all records.

    This is used to make the dashboard look like a real industrial monitoring system.
    """

    df = final_machine_results.copy()
    machines = create_simulated_machines()

    machine_ids = []
    equipment_categories = []

    for i in range(len(df)):
        machine = machines[i % len(machines)]
        machine_ids.append(machine["machine_id"])
        equipment_categories.append(machine["equipment_category"])

    if "machine_id" not in df.columns:
        df.insert(0, "machine_id", machine_ids)

    if "equipment_category" not in df.columns:
        df.insert(1, "equipment_category", equipment_categories)

    print("\nSimulated machine IDs assigned successfully.")
    print(f"Total simulated machines: {len(machines)}")
    print(f"Total equipment categories: {len(set(equipment_categories))}")

    return df


def save_machine_results(machine_results, output_folder="outputs"):
    """
    Save machine-level final results.
    """

    os.makedirs(output_folder, exist_ok=True)

    output_path = os.path.join(output_folder, "machine_results.csv")
    machine_results.to_csv(output_path, index=False)

    print(f"Machine results saved to: {output_path}")