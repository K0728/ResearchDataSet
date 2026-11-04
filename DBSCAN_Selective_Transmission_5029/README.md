# DBSCAN Selective Transmission — Original 5,029-sample vibration dataset

## Before running
Copy your original `Anomaly_Detection.xlsx` into `data/`. This project does not create synthetic data.

## Setup in VS Code
Open this folder in VS Code, then open **Terminal → New Terminal** and run:

```bash
python -m pip install -r requirements.txt
python main.py
```

If `python` is not recognized on Windows, try `py -m pip install -r requirements.txt` and `py main.py`.

## Expected source columns
- `Z_HPF`: signal feature used by DBSCAN
- `Anomali_Flagi`: ground-truth anomaly labels

The script stops if the file or columns are missing. It standardizes `Z_HPF` and uses DBSCAN with `eps=0.12`, `min_samples=10`.

## Outputs in `results/`
- `all_observations_with_dbscan.csv`
- `selected_observations_to_transmit.csv`
- `energy_comparison.csv`
- `summary_results.json`
- `transmission_comparison.png`
- `energy_comparison.png`

## Energy model and limitations
- The full-transmission reference is 30,580 J from the original paper.
- Selective communication energy is estimated proportionally to the selected sample count. This is transparent but does not fully model packet overhead, radio timing, or link conditions.
- Computation energy is estimated as configured processing power × median measured software runtime.
- `PROCESSING_POWER_W = 1.5` is an illustrative assumption, **not a measured device value**. Change it only when you have a justified measurement.
- Full-transmission baseline assumes no filtering computation. Reported savings are estimates, not hardware validation. Do not present them as measured energy savings.
- The code reports actual metrics against the original labels; it does not force a target percentage.
