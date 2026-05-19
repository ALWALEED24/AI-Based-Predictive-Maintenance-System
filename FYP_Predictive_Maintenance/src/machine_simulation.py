import os


EXPECTED_MACHINE_COUNT = 200
EXPECTED_CATEGORY_COUNT = 18


def create_simulated_machines():
    """
    Create simulated industrial machine assets for dashboard monitoring
    and maintenance visualization.

    Total simulated machines: 200
    Total equipment categories: 18

    These equipment categories are simulated operational representations
    created for the SmartMaint AI prototype dashboard.

    They are NOT original equipment labels from the AI4I 2020 dataset.

    The machine distribution is intentionally uneven to better represent
    realistic industrial environments, where equipment quantities differ
    between operational systems and facilities.
    """

    machine_categories = {
        "Motor": {
            "prefix": "MOT",
            "count": 24
        },
        "Pump": {
            "prefix": "PMP",
            "count": 22
        },
        "Compressor": {
            "prefix": "CMP",
            "count": 18
        },
        "Gearbox": {
            "prefix": "GRB",
            "count": 14
        },
        "Conveyor": {
            "prefix": "CNV",
            "count": 14
        },
        "Fan": {
            "prefix": "FAN",
            "count": 12
        },
        "Blower": {
            "prefix": "BLW",
            "count": 10
        },
        "Hydraulic System": {
            "prefix": "HYD",
            "count": 12
        },
        "Turbine": {
            "prefix": "TRB",
            "count": 8
        },
        "Generator": {
            "prefix": "GEN",
            "count": 10
        },
        "Chiller": {
            "prefix": "CHL",
            "count": 10
        },
        "Heat Exchanger": {
            "prefix": "HEX",
            "count": 10
        },
        "Cooling Tower": {
            "prefix": "CLT",
            "count": 8
        },
        "Centrifuge": {
            "prefix": "CFG",
            "count": 8
        },
        "Mixer": {
            "prefix": "MIX",
            "count": 8
        },
        "Agitator": {
            "prefix": "AGT",
            "count": 6
        },
        "Vacuum Pump": {
            "prefix": "VPM",
            "count": 4
        },
        "Separator": {
            "prefix": "SEP",
            "count": 2
        }
    }

    machines = []

    for category, details in machine_categories.items():
        prefix = details["prefix"]
        count = details["count"]

        for i in range(1, count + 1):
            machines.append({
                "machine_id": f"{prefix}-{i:03d}",
                "equipment_category": category
            })

    total_machines = len(machines)
    total_categories = len(machine_categories)

    if total_machines != EXPECTED_MACHINE_COUNT:
        raise ValueError(
            f"Machine simulation error: expected {EXPECTED_MACHINE_COUNT} machines, "
            f"but created {total_machines} machines."
        )

    if total_categories != EXPECTED_CATEGORY_COUNT:
        raise ValueError(
            f"Machine simulation error: expected {EXPECTED_CATEGORY_COUNT} categories, "
            f"but created {total_categories} categories."
        )

    return machines


def assign_machines_to_records(final_machine_results):
    """
    Assign simulated machine IDs and equipment categories
    to monitoring records.

    The assignment repeats cyclically across dataset records
    to simulate a multi-machine industrial monitoring environment.
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
    else:
        df["machine_id"] = machine_ids

    if "equipment_category" not in df.columns:
        df.insert(1, "equipment_category", equipment_categories)
    else:
        df["equipment_category"] = equipment_categories

    unique_machine_count = df["machine_id"].nunique()
    unique_category_count = df["equipment_category"].nunique()

    print("\nSimulated industrial machine assignment completed successfully.")
    print(f"Total simulated machines configured: {len(machines)}")
    print(f"Total equipment categories configured: {EXPECTED_CATEGORY_COUNT}")
    print(f"Unique machines assigned in output: {unique_machine_count}")
    print(f"Unique categories assigned in output: {unique_category_count}")

    if len(df) >= EXPECTED_MACHINE_COUNT and unique_machine_count != EXPECTED_MACHINE_COUNT:
        raise ValueError(
            f"Machine assignment error: expected {EXPECTED_MACHINE_COUNT} unique machines "
            f"in output, but found {unique_machine_count}."
        )

    if len(df) >= EXPECTED_MACHINE_COUNT and unique_category_count != EXPECTED_CATEGORY_COUNT:
        raise ValueError(
            f"Machine assignment error: expected {EXPECTED_CATEGORY_COUNT} unique categories "
            f"in output, but found {unique_category_count}."
        )

    return df


def save_machine_results(machine_results, output_folder="outputs"):
    """
    Save final machine monitoring results.
    """

    os.makedirs(output_folder, exist_ok=True)

    output_path = os.path.join(output_folder, "machine_results.csv")
    machine_results.to_csv(output_path, index=False)

    print(f"Machine monitoring results saved to: {output_path}")