
from pathlib import Path
import time

import numpy as np
import pandas as pd

from sklearn.cluster import DBSCAN
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import precision_score, recall_score, f1_score


ROOT = Path(__file__).resolve().parent
DATA = ROOT / "data" / "Anomaly_Detection.xlsx"
RESULTS = ROOT / "results"
RESULTS.mkdir(exist_ok=True)

# Energy model assumptions
FULL_COMM_J = 30580.0
PROCESSING_POWER_W = 1.5
IDLE_POWER_W = 0.0

# Statistical validation settings
REPEATS = 30

SCENARIOS = {
    "Fixed DBSCAN": {
        "eps": 0.12,
        "min_samples": 10
    },
    "Optimized DBSCAN": {
        "eps": 0.50,
        "min_samples": 30
    }
}


def convert_labels(series):

    numeric = pd.to_numeric(series, errors="coerce")

    if numeric.notna().all() and set(
        numeric.unique()
    ).issubset({0, 1}):

        return numeric.astype(int).to_numpy()

    s = series.astype(str).str.strip().str.lower()

    positive = (
        s.str.contains(
            r"anomal|abnormal|fault|outlier",
            regex=True
        )
        | s.isin(["1", "true", "yes"])
    )

    negative = (
        s.str.contains("normal", regex=False)
        | s.isin(["0", "false", "no"])
    )

    if not (positive | negative).all():
        raise ValueError(
            "Unknown labels found in Anomali_Flagi."
        )

    return positive.astype(int).to_numpy()


def calculate_metrics(y, predicted):

    return {
        "anomaly_precision": precision_score(
            y, predicted, zero_division=0
        ),
        "anomaly_recall": recall_score(
            y, predicted, zero_division=0
        ),
        "anomaly_f1": f1_score(
            y, predicted, zero_division=0
        )
    }


def main():

    if not DATA.exists():
        raise FileNotFoundError(
            f"Dataset not found: {DATA}"
        )

    df = pd.read_excel(DATA)

    required = {"Z_HPF", "Anomali_Flagi"}

    if not required.issubset(df.columns):
        raise ValueError(
            f"Required columns: {required}"
        )

    signal = pd.to_numeric(
        df["Z_HPF"], errors="coerce"
    )

    valid = signal.notna() & df["Anomali_Flagi"].notna()

    df = df.loc[valid].copy()

    y = convert_labels(df["Anomali_Flagi"])

    X = signal.loc[valid].to_numpy().reshape(-1, 1)
    X = StandardScaler().fit_transform(X)

    n = len(y)
    comm_per_sample = FULL_COMM_J / n

    all_runs = []

    # Full transmission reference
    all_runs.append({
        "scenario": "Full transmission",
        "run": 0,
        "eps": np.nan,
        "min_samples": np.nan,
        "samples_transmitted": n,
        "runtime_seconds": 0.0,
        "computation_energy_j": 0.0,
        "communication_energy_j": FULL_COMM_J,
        "total_energy_j": FULL_COMM_J,
        "anomaly_precision": np.nan,
        "anomaly_recall": np.nan,
        "anomaly_f1": np.nan
    })

    # Repeat each DBSCAN scenario
    for scenario, params in SCENARIOS.items():

        for run in range(1, REPEATS + 1):

            start = time.perf_counter()

            labels = DBSCAN(
                eps=params["eps"],
                min_samples=params["min_samples"]
            ).fit_predict(X)

            runtime = time.perf_counter() - start

            # Noise points are selected for transmission
            selected = (labels == -1).astype(int)

            count = int(selected.sum())

            # Energy estimates
            computation_energy = (
                max(
                    PROCESSING_POWER_W - IDLE_POWER_W,
                    0.0
                ) * runtime
            )

            communication_energy = (
                count * comm_per_sample
            )

            total_energy = (
                computation_energy
                + communication_energy
            )

            metrics = calculate_metrics(y, selected)

            all_runs.append({
                "scenario": scenario,
                "run": run,
                "eps": params["eps"],
                "min_samples": params["min_samples"],
                "samples_transmitted": count,
                "runtime_seconds": runtime,
                "computation_energy_j": computation_energy,
                "communication_energy_j": communication_energy,
                "total_energy_j": total_energy,
                **metrics
            })

    # Save individual run results
    runs_df = pd.DataFrame(all_runs)

    runs_file = RESULTS / "repeated_runs_all.csv"

    runs_df.to_csv(
        runs_file,
        index=False
    )

    # Calculate statistical summary
    summary_rows = []

    for scenario, group in runs_df.groupby("scenario"):

        row = {"scenario": scenario}

        for column in [
            "samples_transmitted",
            "runtime_seconds",
            "computation_energy_j",
            "communication_energy_j",
            "total_energy_j",
            "anomaly_precision",
            "anomaly_recall",
            "anomaly_f1"
        ]:

            values = group[column].dropna()

            if len(values) == 0:
                row[f"{column}_mean"] = np.nan
                row[f"{column}_std"] = np.nan
                row[f"{column}_ci95_lower"] = np.nan
                row[f"{column}_ci95_upper"] = np.nan
                continue

            mean_value = values.mean()
            std_value = values.std(ddof=1)

            # Approximate 95% confidence interval
            margin = (
                1.96 * std_value / np.sqrt(len(values))
            )

            row[f"{column}_mean"] = mean_value
            row[f"{column}_std"] = std_value
            row[f"{column}_ci95_lower"] = (
                mean_value - margin
            )
            row[f"{column}_ci95_upper"] = (
                mean_value + margin
            )

        summary_rows.append(row)

    summary_df = pd.DataFrame(summary_rows)

    summary_file = RESULTS / "repeated_runs_summary.csv"

    summary_df.to_csv(
        summary_file,
        index=False
    )

    print("\n=== REPEATED-RUN VALIDATION COMPLETE ===")
    print("Dataset rows:", n)
    print("DBSCAN repetitions per scenario:", REPEATS)

    print("\nStatistical summary:")
    print(summary_df.to_string(index=False))

    print("\nSaved:")
    print(runs_file)
    print(summary_file)

    print(
        "\nNOTE: All energy values are simulation-based estimates."
    )


if __name__ == "__main__":
    main()
