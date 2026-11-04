"""
Fair three-scenario comparison on the ORIGINAL Anomaly_Detection.xlsx dataset.
Creates a separate report and does not overwrite main.py / optimization.py outputs.

Scenarios:
1) Full transmission: transmit all valid observations; no DBSCAN computation.
2) Fixed DBSCAN: eps=0.12, min_samples=10.
3) Optimized DBSCAN: eps=0.50, min_samples=30 (selected in previous search with recall >= 0.95).

Energy model is intentionally explicit:
- Full communication reference = 30,580 J from the paper.
- For DBSCAN scenarios, communication energy is scaled linearly by selected sample count.
- Computation energy = (assumed processing power - assumed idle power) * median runtime.
- These are model estimates, NOT hardware measurements. The radio model does not account for packet overhead, timing, retries, or link conditions.
"""

from pathlib import Path
import json
import time

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.cluster import DBSCAN
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import precision_score, recall_score, f1_score, confusion_matrix

ROOT = Path(__file__).resolve().parent
DATA = ROOT / "data" / "Anomaly_Detection.xlsx"
OUT = ROOT / "results"
OUT.mkdir(exist_ok=True)

SIGNAL_COL = "Z_HPF"
LABEL_COL = "Anomali_Flagi"
FULL_COMM_J = 30580.0
PROCESSING_POWER_W = 1.5
IDLE_POWER_W = 0.0
POWER_IS_MEASURED = False
REPEATS = 10

SCENARIOS = [
    {"name": "Full transmission", "mode": "full", "eps": None, "min_samples": None},
    {"name": "Fixed DBSCAN", "mode": "dbscan", "eps": 0.12, "min_samples": 10},
    {"name": "Optimized DBSCAN", "mode": "dbscan", "eps": 0.50, "min_samples": 30},
]


def map_labels(series):
    numeric = pd.to_numeric(series, errors="coerce")
    if numeric.notna().all() and set(numeric.unique()).issubset({0, 1}):
        return numeric.astype(int).to_numpy()
    s = series.astype(str).str.strip().str.lower()
    pos = s.str.contains(r"anomal|abnormal|fault|outlier", regex=True) | s.isin({"1", "true", "yes"})
    neg = s.str.contains("normal", regex=False) | s.isin({"0", "false", "no"})
    if not (pos | neg).all():
        raise ValueError(f"Unrecognized label values in {LABEL_COL}: {series.unique()[:10]}")
    return pos.astype(int).to_numpy()


def main():
    if not DATA.exists():
        raise FileNotFoundError(f"Original dataset missing: {DATA}. No synthetic data will be generated.")

    df = pd.read_excel(DATA)
    missing = {SIGNAL_COL, LABEL_COL} - set(df.columns)
    if missing:
        raise ValueError(f"Missing columns {missing}; available columns: {list(df.columns)}")

    signal = pd.to_numeric(df[SIGNAL_COL], errors="coerce")
    valid = signal.notna() & df[LABEL_COL].notna()
    df = df.loc[valid].copy().reset_index(drop=True)
    X = signal.loc[valid].to_numpy().reshape(-1, 1)
    y = map_labels(df[LABEL_COL])
    X_scaled = StandardScaler().fit_transform(X)
    n = len(df)
    comm_per_observation = FULL_COMM_J / n

    rows = []
    for sc in SCENARIOS:
        if sc["mode"] == "full":
            selected = np.ones(n, dtype=int)
            runtime = 0.0
            y_pred = np.ones(n, dtype=int)  # all observations transmitted; not an anomaly classifier
            # Quality metrics are not applicable to full transmission as a selection/anomaly detector.
            precision = recall = f1 = np.nan
            selected_count = n
            e_comp = 0.0
            e_comm = FULL_COMM_J
        else:
            runtimes = []
            labels = None
            for _ in range(REPEATS):
                start = time.perf_counter()
                labels = DBSCAN(eps=sc["eps"], min_samples=sc["min_samples"]).fit_predict(X_scaled)
                runtimes.append(time.perf_counter() - start)
            runtime = float(np.median(runtimes))
            selected = (labels == -1).astype(int)
            selected_count = int(selected.sum())
            precision = precision_score(y, selected, zero_division=0)
            recall = recall_score(y, selected, zero_division=0)
            f1 = f1_score(y, selected, zero_division=0)
            e_comp = max(0.0, PROCESSING_POWER_W - IDLE_POWER_W) * runtime
            e_comm = selected_count * comm_per_observation

        e_total = e_comp + e_comm
        saving_j = FULL_COMM_J - e_total
        saving_pct = 100 * saving_j / FULL_COMM_J if FULL_COMM_J else 0.0

        rows.append({
            "scenario": sc["name"],
            "eps": sc["eps"],
            "min_samples": sc["min_samples"],
            "dataset_rows": n,
            "samples_transmitted": selected_count,
            "data_reduction_percent": 100 * (1 - selected_count / n),
            "median_filter_runtime_seconds": runtime,
            "anomaly_precision": precision,
            "anomaly_recall": recall,
            "anomaly_f1": f1,
            "computation_energy_j_estimate": e_comp,
            "communication_energy_j_estimate": e_comm,
            "total_energy_j_estimate": e_total,
            "saving_vs_full_transmission_percent_estimate": saving_pct,
            "processing_power_w_assumed": PROCESSING_POWER_W,
            "power_is_measured": POWER_IS_MEASURED,
        })

    result = pd.DataFrame(rows)
    result.to_csv(OUT / "three_scenario_comparison.csv", index=False)

    # Compare DBSCAN scenarios against fixed baseline too.
    fixed_total = float(result.loc[result["scenario"] == "Fixed DBSCAN", "total_energy_j_estimate"].iloc[0])
    result["saving_vs_fixed_dbscan_percent_estimate"] = (
        100 * (fixed_total - result["total_energy_j_estimate"]) / fixed_total
    )
    result.to_csv(OUT / "three_scenario_comparison.csv", index=False)

    summary = {
        "dataset": DATA.name,
        "rows_used": n,
        "ground_truth_normal": int((y == 0).sum()),
        "ground_truth_anomaly": int((y == 1).sum()),
        "scenarios": result.replace({np.nan: None}).to_dict(orient="records"),
        "energy_assumptions": {
            "full_transmission_communication_reference_j": FULL_COMM_J,
            "processing_power_w_assumed": PROCESSING_POWER_W,
            "idle_power_w_assumed": IDLE_POWER_W,
            "power_is_measured": POWER_IS_MEASURED,
            "communication_method": "Linear scaling of full-transmission reference by sample count.",
            "computation_method": "(assumed processing power - assumed idle power) * median runtime.",
            "limitations": "Simulation estimate only; no hardware measurement or packet-level radio model."
        }
    }
    with open(OUT / "three_scenario_summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, allow_nan=False)

    # Energy plot
    plt.figure()
    plt.bar(result["scenario"], result["total_energy_j_estimate"])
    plt.ylabel("Estimated total energy (J)")
    plt.title("Three-Scenario Energy Comparison (Estimated)")
    plt.xticks(rotation=12, ha="right")
    plt.tight_layout()
    plt.savefig(OUT / "three_scenario_energy.png", dpi=200)
    plt.close()

    print("\n=== THREE-SCENARIO COMPARISON COMPLETE ===")
    print(result[[
        "scenario", "samples_transmitted", "data_reduction_percent",
        "anomaly_precision", "anomaly_recall", "anomaly_f1",
        "computation_energy_j_estimate", "communication_energy_j_estimate",
        "total_energy_j_estimate", "saving_vs_full_transmission_percent_estimate",
        "saving_vs_fixed_dbscan_percent_estimate"
    ]].to_string(index=False))
    print("\nSaved:")
    print("results/three_scenario_comparison.csv")
    print("results/three_scenario_summary.json")
    print("results/three_scenario_energy.png")
    print("\nWARNING: all energy values are estimates, not measured hardware energy.")


if __name__ == "__main__":
    main()
