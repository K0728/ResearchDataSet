
from pathlib import Path
import pandas as pd

# Project directory
ROOT = Path(__file__).resolve().parent
RESULTS = ROOT / "results"

# Input and output files
INPUT_FILE = RESULTS / "dbscan_parameter_search.csv"
SUMMARY_FILE = RESULTS / "sensitivity_summary.csv"
ALL_CONFIGS_FILE = RESULTS / "sensitivity_all_configs.csv"

# Energy model assumptions
FULL_COMM_J = 30580.0

# Processing power scenarios (Watts)
POWER_SCENARIOS_W = [0.5, 1.5, 3.0, 5.0]

# Idle power assumption (Watts)
IDLE_POWER_W = 0.0

# Minimum acceptable anomaly recall
MIN_RECALL = 0.95


def main():

    # Check input file
    if not INPUT_FILE.exists():
        raise FileNotFoundError(
            f"\nInput CSV not found:\n{INPUT_FILE}\n"
            "Run optimization.py first to generate the CSV."
        )

    # Load optimization results
    df = pd.read_csv(INPUT_FILE)

    # Validate required columns
    required_columns = {
        "eps",
        "min_samples",
        "selected_samples",
        "data_reduction_percent",
        "anomaly_recall",
        "anomaly_f1",
        "runtime_seconds",
        "communication_energy_j_estimate",
    }

    missing = required_columns - set(df.columns)

    if missing:
        raise ValueError(
            f"Missing columns in input CSV: {missing}"
        )

    if df.empty:
        raise ValueError("Input CSV contains no results.")

    all_results = []
    summaries = []

    # Evaluate each processing power scenario
    for power in POWER_SCENARIOS_W:

        results = df.copy()

        # Effective processing power after idle power
        effective_power = max(
            power - IDLE_POWER_W, 0.0
        )

        # Recalculate computation energy
        results["computation_energy_j_estimate"] = (
            effective_power * results["runtime_seconds"]
        )

        # Recalculate total energy
        results["total_energy_j_estimate"] = (
            results["communication_energy_j_estimate"]
            + results["computation_energy_j_estimate"]
        )

        # Recalculate energy saving
        results["estimated_energy_saving_percent"] = (
            100 * (
                FULL_COMM_J - results["total_energy_j_estimate"]
            ) / FULL_COMM_J
        )

        # Store scenario information
        results["processing_power_w"] = power
        results["idle_power_w"] = IDLE_POWER_W

        # Apply recall constraint
        eligible = results[
            results["anomaly_recall"] >= MIN_RECALL
        ]

        if eligible.empty:

            summaries.append({
                "processing_power_w": power,
                "status": "No eligible configuration"
            })

            continue

        # Select configuration with minimum total energy
        best = eligible.sort_values(
            ["total_energy_j_estimate", "anomaly_f1"],
            ascending=[True, False]
        ).iloc[0]

        summaries.append({
            "processing_power_w": power,
            "eps": best["eps"],
            "min_samples": int(best["min_samples"]),
            "selected_samples": int(best["selected_samples"]),
            "data_reduction_percent":
                best["data_reduction_percent"],
            "anomaly_precision":
                best["anomaly_precision"],
            "anomaly_recall":
                best["anomaly_recall"],
            "anomaly_f1":
                best["anomaly_f1"],
            "runtime_seconds":
                best["runtime_seconds"],
            "computation_energy_j":
                best["computation_energy_j_estimate"],
            "communication_energy_j":
                best["communication_energy_j_estimate"],
            "total_energy_j":
                best["total_energy_j_estimate"],
            "energy_saving_percent":
                best["estimated_energy_saving_percent"],
            "status": "Eligible"
        })

        all_results.append(results)

    # Save summary results
    summary_df = pd.DataFrame(summaries)

    summary_df.to_csv(
        SUMMARY_FILE,
        index=False
    )

    # Save all configurations for every scenario
    if all_results:

        pd.concat(
            all_results,
            ignore_index=True
        ).to_csv(
            ALL_CONFIGS_FILE,
            index=False
        )

    # Display results
    print("\n=== SENSITIVITY ANALYSIS COMPLETE ===")

    print("\nProcessing power scenarios:")
    print(POWER_SCENARIOS_W)

    print("\nMinimum recall constraint:")
    print(MIN_RECALL)

    print("\nResults:")
    print(summary_df.to_string(index=False))

    print("\nSaved files:")
    print(SUMMARY_FILE)
    print(ALL_CONFIGS_FILE)

    print(
        "\nNOTE: All energy values are simulation-based estimates, "
        "not hardware measurements."
    )


if __name__ == "__main__":
    main()
