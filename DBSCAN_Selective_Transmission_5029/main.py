"""DBSCAN selective sensor-data transmission using the original vibration dataset.
No synthetic samples are generated. See README.md for energy-model limitations.
"""
from pathlib import Path
import time, json, warnings
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import DBSCAN
from sklearn.metrics import confusion_matrix, classification_report

ROOT = Path(__file__).resolve().parent
DATA_PATH = ROOT / 'data' / 'Anomaly_Detection.xlsx'
OUT = ROOT / 'results'
OUT.mkdir(exist_ok=True)
SIGNAL_COL = 'Z_HPF'
LABEL_COL = 'Anomali_Flagi'
EPS = 0.12
MIN_SAMPLES = 10
FULL_TRANSMISSION_ENERGY_J = 30580.0  # original paper reference
PROCESSING_POWER_W = 1.5              # illustrative assumption; replace with measured power
PROCESSING_POWER_IS_MEASURED = False
RUNTIME_REPEATS = 10


def binary_labels(series):
    nums = pd.to_numeric(series, errors='coerce')
    if nums.notna().all() and set(nums.unique()).issubset({0, 1}):
        return nums.astype(int).to_numpy()
    s = series.astype(str).str.strip().str.lower()
    positive = s.str.contains(r'anomal|abnormal|fault|outlier', regex=True) | s.isin({'1','true','yes'})
    negative = s.str.contains('normal', regex=False) | s.isin({'0','false','no'})
    if not (positive | negative).all():
        raise ValueError(f'Cannot map labels safely. Unique source values: {series.unique()[:20]}')
    return positive.astype(int).to_numpy()


def main():
    if not DATA_PATH.exists():
        raise FileNotFoundError(f'Original dataset missing: {DATA_PATH}\nCopy your original Anomaly_Detection.xlsx into data/.')
    df = pd.read_excel(DATA_PATH)
    missing = {SIGNAL_COL, LABEL_COL} - set(df.columns)
    if missing:
        raise ValueError(f'Missing columns {missing}. Available columns: {list(df.columns)}')
    original_n = len(df)
    signal = pd.to_numeric(df[SIGNAL_COL], errors='coerce')
    valid = signal.notna() & df[LABEL_COL].notna()
    if not valid.all():
        warnings.warn(f'Excluding {(~valid).sum()} rows with missing signal/label values.')
    work = df.loc[valid].copy().reset_index().rename(columns={'index':'original_row_index'})
    X = work[SIGNAL_COL].astype(float).to_numpy().reshape(-1, 1)
    y = binary_labels(work[LABEL_COL])
    Xs = StandardScaler().fit_transform(X)

    runtimes, cluster_labels = [], None
    for _ in range(RUNTIME_REPEATS):
        start = time.perf_counter()
        cluster_labels = DBSCAN(eps=EPS, min_samples=MIN_SAMPLES).fit_predict(Xs)
        runtimes.append(time.perf_counter() - start)
    runtime_s = float(np.median(runtimes))
    pred = (cluster_labels == -1).astype(int)
    selected = pred == 1
    n = len(work)
    selected_n = int(selected.sum())
    reduction_pct = (1 - selected_n / n) * 100 if n else 0.0
    cm = confusion_matrix(y, pred, labels=[0, 1])
    report = classification_report(y, pred, labels=[0,1], target_names=['normal','anomaly'], output_dict=True, zero_division=0)

    # Transparent estimates, not hardware measurements.
    comm_per_sample_j = FULL_TRANSMISSION_ENERGY_J / n
    e_comm_full = FULL_TRANSMISSION_ENERGY_J
    e_comm_selected = selected_n * comm_per_sample_j
    e_comp = PROCESSING_POWER_W * runtime_s
    e_total_full = e_comm_full
    e_total_selective = e_comm_selected + e_comp
    saving_j = e_total_full - e_total_selective
    saving_pct = 100 * saving_j / e_total_full if e_total_full else 0.0

    work['dbscan_cluster'] = cluster_labels
    work['ground_truth_anomaly'] = y
    work['predicted_anomaly'] = pred
    work['selected_for_transmission'] = selected
    work.to_csv(OUT/'all_observations_with_dbscan.csv', index=False)
    work.loc[selected].to_csv(OUT/'selected_observations_to_transmit.csv', index=False)
    energy = pd.DataFrame([
        {'scenario':'Full transmission (no DBSCAN)', 'samples_transmitted':n, 'data_reduction_percent':0.0, 'computation_energy_j':0.0, 'communication_energy_j_estimate':e_comm_full, 'total_energy_j_estimate':e_total_full},
        {'scenario':'DBSCAN selective transmission', 'samples_transmitted':selected_n, 'data_reduction_percent':reduction_pct, 'computation_energy_j':e_comp, 'communication_energy_j_estimate':e_comm_selected, 'total_energy_j_estimate':e_total_selective},
    ])
    energy.to_csv(OUT/'energy_comparison.csv', index=False)
    summary = {
        'dataset_file':DATA_PATH.name, 'original_rows_in_file':int(original_n), 'rows_used':int(n),
        'signal_column':SIGNAL_COL, 'ground_truth_column':LABEL_COL, 'eps':EPS, 'min_samples':MIN_SAMPLES,
        'runtime_repeats':RUNTIME_REPEATS, 'median_dbscan_runtime_seconds':runtime_s,
        'normal_ground_truth':int((y==0).sum()), 'anomaly_ground_truth':int((y==1).sum()),
        'selected_for_transmission':selected_n, 'data_reduction_percent':reduction_pct,
        'confusion_matrix_order':['normal','anomaly'], 'confusion_matrix':cm.tolist(), 'classification_report':report,
        'energy_model':{'full_transmission_reference_j':e_comm_full,'communication_energy_per_observation_estimate_j':comm_per_sample_j,
          'selective_communication_energy_estimate_j':e_comm_selected,'processing_power_w':PROCESSING_POWER_W,
          'processing_power_is_measured':PROCESSING_POWER_IS_MEASURED,'computation_energy_estimate_j':e_comp,
          'full_transmission_total_energy_estimate_j':e_total_full,'selective_total_energy_estimate_j':e_total_selective,
          'estimated_energy_saved_j':saving_j,'estimated_total_energy_saving_percent':saving_pct,
          'warning':'Communication energy is scaled by sample count from the paper reference. Computation energy is runtime × configured power. Estimates only; not hardware validation.'}}
    (OUT/'summary_results.json').write_text(json.dumps(summary, indent=2), encoding='utf-8')
    plt.figure(); plt.bar(['Full transmission','DBSCAN selective'], [n, selected_n]); plt.ylabel('Observations transmitted'); plt.title('Selective Sensor-Data Transmission'); plt.tight_layout(); plt.savefig(OUT/'transmission_comparison.png', dpi=200); plt.close()
    plt.figure(); plt.bar(['Full transmission','DBSCAN selective'], [e_total_full, e_total_selective]); plt.ylabel('Estimated energy (J)'); plt.title('Total Energy Comparison (estimate)'); plt.tight_layout(); plt.savefig(OUT/'energy_comparison.png', dpi=200); plt.close()
    print('\n=== ORIGINAL DATASET RESULTS ===')
    print(f'Rows in source: {original_n}; rows used: {n}')
    print(f'Ground-truth anomalies: {(y==1).sum()}')
    print(f'Selected for transmission: {selected_n}; data reduction: {reduction_pct:.2f}%')
    print(f'Median DBSCAN runtime ({RUNTIME_REPEATS} runs): {runtime_s:.6f} s')
    print('Confusion matrix [normal, anomaly]:\n', cm)
    print(f'Estimated computation energy: {e_comp:.6f} J')
    print(f'Estimated communication energy — full: {e_comm_full:.6f} J; selective: {e_comm_selected:.6f} J')
    print(f'Estimated total energy — full: {e_total_full:.6f} J; selective: {e_total_selective:.6f} J')
    print(f'Estimated total-energy saving: {saving_pct:.2f}%')
    print('Outputs saved to:', OUT)
    print('NOTE: energy values are estimates until power is measured on the target device.')

if __name__ == '__main__':
    main()
