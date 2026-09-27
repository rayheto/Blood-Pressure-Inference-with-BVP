# PPG-only rise correction on the matched VitalDB cohort

This is the ECG ablation of the [PPG + ECG rise-aware experiment](../rise_head/README.md). Both runs use the **same** matched VitalDB windows, 51,991 first-calibration training pairs, 446 validation pairs, 784 held-out test pairs, subject split, seed 42, loss, rise-bin/case sampling, validation-selection rule, and 30-minute horizon. Each rise head uses its own frozen base encoder. The PPG-only base has no ECG, PAT, or RR input; `StreamingBloodPressure` accepts `ecg=None` for calibration and observation in this mode. The PPG-only head has 69 state inputs and was selected at epoch 59; the fusion head has 73 and was selected at epoch 22.

## Numeric comparison

MAE is in mmHg, calculated on **unsmoothed** BP-change predictions.

| Dataset / model | SBP MAE | DBP MAE | SBP MAE for rise ≥40 |
|---|---:|---:|---:|
| Validation: PPG base | 18.04 | 8.25 | 38.27 (70 windows) |
| Validation: PPG rise head | **11.99** | **5.24** | **19.57** (70 windows) |
| Validation: fusion rise head | 12.74 | 5.83 | 24.06 (70 windows) |
| Held out: PPG base | 14.26 | **6.13** | 11.65 (38 windows) |
| Held out: PPG rise head | **10.14** | **5.61** | 9.11 (38 windows) |
| Held out: fusion rise head | 11.00 | 6.54 | **7.92** (38 windows) |

For the 32 held-out cases, PPG-only rise head minus fusion rise head overall MAE is **-0.86 SBP / -0.93 DBP**. Paired case-bootstrap 95% intervals are **[-2.39, +0.42] SBP** and **[-1.63, -0.07] DBP**. Thus the SBP difference is uncertain; the DBP interval favors PPG-only on this repeatedly inspected test set. For rise≥40, PPG-only SBP MAE is **1.19 mmHg higher**, with a case-bootstrap interval for the difference of **[-1.07, +6.25]**. These 38 windows come from only eight cases.

A separate replay of original continuous held-out VitalDB case 236 produced 175 valid 10-second windows:

| Model | SBP MAE | DBP MAE |
|---|---:|---:|
| PPG base | 29.29 | 15.05 |
| PPG-only rise head | 12.36 | 8.64 |
| PPG + ECG rise head | **8.07** | **5.46** |

Within this case's 15 windows at reference SBP 160–180 mmHg, PPG-only versus fusion SBP MAE is **41.51 vs 26.98 mmHg**. This case's earlier curve helped motivate the rise-head approach, so it is exploratory evidence, not a fresh independent test. The held-out set has also been inspected repeatedly. A new independent cohort and wrist PPG recordings are needed before asserting the modality ranking or clinical accuracy.

The [animated SVG](case_236_ppg_vs_fusion.svg) uses **four curves**: arterial-pressure reference, original PPG base, PPG-only rise head, and PPG + ECG rise head. All use the same trailing 60-second display average. `case_236_trace.json` contains each unsmoothed prediction and `create_animation.py` regenerates the SVG.

The animation is also available as a [20-second MP4 video](case_236_ppg_vs_fusion.mp4) at 1200×790 and 24 fps.

## Artifacts

- `rise_head.pt`: PPG-only PyTorch correction checkpoint. Pair with `../ppg_only.pt`.
- `metrics.json`: training selection and validation/test metrics.
- `test_case_bootstrap.json`: paired PPG-only versus fusion uncertainty.
- `raw_stream_case_236.json`, `case_236_trace.json`: continuous replay results and sequence.
- `../../train_rise_head.py --mode ppg_only`, `../../evaluate_ppg_rise.py`: reproducible experiment scripts.
- `../../stream_inference.py`: use `mode='ppg_only'`, `rise_checkpoint=...`, and `ecg=None`.

TensorBoard run: `VitalDB_stream_rise_head_ppg_only` under `D:/BP_Training/all_eligible/tensorboard_merged`. These PyTorch weights are not quantized or tested on ESP32-S3. Arterial pressure supplies simulated cuff calibration and test labels in the operating-room data.
