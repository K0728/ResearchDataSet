
from pathlib import Path
import json
import time

import numpy as np
import pandas as pd
from sklearn.cluster import DBSCAN
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import precision_score, recall_score, f1_score

ROOT = Path(__file__).resolve().parent
DATA = ROOT / "data" / "Anomaly_Detection.xlsx"
OUT = ROOT / "results"
OUT.mkdir(exist_ok=True)

# Original paper baseline
EPS_VALUES = [0.03, 0.05, 0.07, 0.09, 0.12, 0.15,
              0.18, 0.20, 0.25, 0.30, 0.40, 0.50]
MIN_SAMPLES_VALUES = [5, 8, 10, 15, 20, 30, 50]

FULL_COMM_J = 30580.0
PROCESSING_POWER_W = 1.5  # Assumption, not measured hardware power
IDLE_POWER_W = 0.0        # Assumption; replace with measurement
MIN_RECALL = 0.95
REPEATS = 30


def convert_labels(series):
    numeric = pd.to_numeric(series, errors="coerce")

    if numeric.notna().all() and set(numeric.unique()).issubset({0, 1}):
        return numeric.astype(int).to_numpy()

    s = series.astype(str).str.strip().str.lower()
    positive = (
        s.str.contains(r"anomal|abnormal|fault|outlier", regex=True)
        | s.isin(["1", "true", "yes"])
    )
    negative = (
        s.str.contains("normal", regex=False)
        | s.isin(["0", "false", "no"])
    )

    if not (positive | negative).all():
        raise ValueError("Unknown labels found in Anomali_Flagi.")

    return positive.astype(int).to_numpy()


def evaluate(X, y, eps, min_samples):
    times = []

    for _ in range(REPEATS):
        start = time.perf_counter()
        labels = DBSCAN(
            eps=eps, min_samples=min_samples
        ).fit_predict(X)
        times.append(time.perf_counter() - start)

    runtime = float(np.median(times))

    # DBSCAN noise points are selected for transmission.
    selected = (labels == -1).astype(int)
    count = int(selected.sum())
    n = len(y)

    precision = precision_score(y, selected, zero_division=0)
    recall = recall_score(y, selected, zero_division=0)
    f1 = f1_score(y, selected, zero_division=0)

    reduction = 100 * (1 - count / n)

    # Transparent energy estimates
    comm_per_sample = FULL_COMM_J / n
    e_comm = count * comm_per_sample
    e_comp = max(0, PROCESSING_POWER_W - IDLE_POWER_W) * runtime
    e_total = e_comm + e_comp

    return {
        "eps": eps,
        "min_samples": min_samples,
        "selected_samples": count,
        "data_reduction_percent": reduction,
        "anomaly_precision": precision,
        "anomaly_recall": recall,
        "anomaly_f1": f1,
        "runtime_seconds": runtime,
        "computation_energy_j_estimate": e_comp,
        "communication_energy_j_estimate": e_comm,
        "total_energy_j_estimate": e_total,
        "estimated_energy_saving_percent":
            100 * (FULL_COMM_J - e_total) / FULL_COMM_J,
    }


def main():
    if not DATA.exists():
        raise FileNotFoundError(
            f"Original dataset not found: {DATA}"
        )

    df = pd.read_excel(DATA)

    required = {"Z_HPF", "Anomali_Flagi"}
    if not required.issubset(df.columns):
        raise ValueError(
            f"Required columns: {required}; found: {list(df.columns)}"
        )

    signal = pd.to_numeric(df["Z_HPF"], errors="coerce")
    valid = signal.notna() & df["Anomali_Flagi"].notna()

    df = df.loc[valid].copy()
    y = convert_labels(df["Anomali_Flagi"])
    X = signal.loc[valid].to_numpy().reshape(-1, 1)
    X = StandardScaler().fit_transform(X)

    rows = []

    for eps in EPS_VALUES:
        for min_samples in MIN_SAMPLES_VALUES:
            result = evaluate(X, y, eps, min_samples)
            rows.append(result)

    results = pd.DataFrame(rows)
    results.to_csv(OUT / "dbscan_parameter_search_30repeats.csv", index=False) 

    eligible = results[results["anomaly_recall"] >= MIN_RECALL]

    if eligible.empty:
        best = None
        message = (
            "No configuration meets the recall constraint. "
            "Review the results; do not force a target saving."
        )
    else:
        best = eligible.sort_values(
            ["total_energy_j_estimate", "anomaly_f1"],
            ascending=[True, False]
        ).iloc[0].to_dict()

        message = "Best tested configuration meeting recall constraint."

    baseline = results[
        np.isclose(results["eps"], 0.12)
        & (results["min_samples"] == 10)
    ].iloc[0].to_dict()

    summary = {
        "dataset": DATA.name,
        "rows_used": len(y),
        "configurations_tested": len(results),
        "minimum_anomaly_recall": MIN_RECALL,
        "baseline_eps_0.12_min_samples_10": baseline,
        "best_eligible_configuration": best,
        "selection_message": message,
        "energy_assumptions": {
            "full_transmission_reference_j": FULL_COMM_J,
            "processing_power_w": PROCESSING_POWER_W,
            "idle_power_w": IDLE_POWER_W,
            "power_measured": False,
            "communication_model":
                "Full-transmission reference scaled linearly by sample count.",
            "computation_model":
                "(processing power - idle power) * median runtime",
            "limitation":
                "Estimated energy only; no hardware measurement or packet-level radio model."
        }
    }

    with open(
        OUT / "optimization_summary.json", "w", encoding="utf-8"
    ) as f:
        json.dump(summary, f, indent=2)

    print("\n=== DBSCAN OPTIMIZATION COMPLETE ===")
    print("Rows used:", len(y))
    print("Configurations tested:", len(results))
    print("\nOriginal baseline: eps=0.12, min_samples=10")
    print("Selected:", int(baseline["selected_samples"]))
    print("Reduction:", round(baseline["data_reduction_percent"], 2), "%")
    print("Precision:", round(baseline["anomaly_precision"], 4))
    print("Recall:", round(baseline["anomaly_recall"], 4))
    print("F1:", round(baseline["anomaly_f1"], 4))

    print("\n", message)

    if best is not None:
        print("Best eps:", best["eps"])
        print("Best min_samples:", int(best["min_samples"]))
        print("Selected:", int(best["selected_samples"]))
        print("Reduction:", round(best["data_reduction_percent"], 2), "%")
        print("Precision:", round(best["anomaly_precision"], 4))
        print("Recall:", round(best["anomaly_recall"], 4))
        print("F1:", round(best["anomaly_f1"], 4))
        print("Estimated total energy:",
              round(best["total_energy_j_estimate"], 6), "J")
        print("Estimated saving:",
              round(best["estimated_energy_saving_percent"], 2), "%")

    print("\nSaved:")
    print("results/dbscan_parameter_search.csv")
    print("results/optimization_summary.json")
    print("\nEnergy values are estimates, not hardware measurements.")


if __name__ == "__main__":
    main()
