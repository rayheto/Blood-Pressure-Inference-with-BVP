# Supervised dynamic correction head

This experiment freezes the existing `ppg_pat_rr.pt` streaming model and trains a small bounded residual head. Its five inputs are cuff-calibration SBP/DBP, the frozen model's predicted SBP/DBP change, and time since calibration. Each output adjustment is bounded to ±60 mmHg SBP and ±30 mmHg DBP. The head does not observe the current arterial-pressure label at inference.

The same 51,991 first-calibration training pairs were used. Checkpoints were selected on 446 validation pairs from separate cases, with an objective combining overall SBP/DBP MAE and SBP MAE when reference SBP was at least 140 mmHg. `uniform.pt` uses a Huber regression loss; `high_pressure_weighted.pt` increases loss weight and underestimation penalty at high reference pressure. Both were trained with seed 42. Training curves are under `VitalDB_stream_dynamic_head` in `D:/BP_Training/all_eligible/tensorboard_merged`.

## Results

The original, uniform, and high-pressure-weighted models were evaluated on the same 32 held-out cases and 784 sparse windows. MAE is in mmHg; lower is better.

| Model | SBP change MAE | DBP change MAE | SBP ≥140 MAE (n=71) | SBP ≥160 MAE (n=9) |
|---|---:|---:|---:|---:|
| Frozen base | 13.71 | 6.26 | 9.67 | 6.51 |
| Uniform correction | **10.41** | 6.28 | 10.77 | 15.72 |
| High-pressure-weighted correction | 11.75 | 6.53 | **8.99** | 7.46 |

Against the base, case-bootstrap 95% intervals for MAE difference were **[-6.04, -0.26] SBP** and **[-0.95, +0.87] DBP** for uniform correction; **[-4.91, +1.13] SBP** and **[-0.52, +1.11] DBP** for high-pressure weighting. An interval below zero indicates lower error. Only uniform correction's overall SBP interval excluded zero. The ≥160 subgroup has just nine windows and does not support a reliable conclusion.

The original continuous recording of held-out case 236 was also replayed for 175 valid 10-second windows after one simulated cuff calibration:

| Model | SBP change MAE | DBP change MAE |
|---|---:|---:|
| Keep calibration value | 17.95 | **9.12** |
| Frozen base | 26.51 | 13.66 |
| Uniform correction | **14.97** | 9.71 |
| High-pressure-weighted correction | 15.27 | 9.59 |

The [animated comparison](case_236_dynamic_fit.svg) applies an identical trailing 60-second average to the reference and all predictions for viewing. The table above is scored on **unsmoothed** windows. `case_236_trace.json` contains the underlying values. Case 236 is in the held-out split, but its curve had been inspected before this correction-head experiment was designed; it is **not a fresh independent confirmation**. The sparse-window test set has also been inspected repeatedly. Confirmation needs new cases and ideally wrist PPG plus simultaneous cuff measurements. No result establishes clinical accuracy.

## Files

- `uniform.pt` and `high_pressure_weighted.pt`: PyTorch correction checkpoints.
- `metrics.json`, `test_case_bootstrap.json`, `raw_stream_case_236.json`: numeric evaluations.
- `case_236_trace.json`, `case_236_dynamic_fit.svg`, `create_animation.py`: replay trace and animation.
- `../../train_dynamic_head.py`, `../../evaluate_dynamic_head.py`: training and case-bootstrap scripts.
- `../../stream_inference.py`: pass `correction_checkpoint=Path('.../uniform.pt')` with `mode='ppg_pat_rr'` to apply the selected head during online inference.

The PyTorch interface has not been quantized or tested on ESP32-S3. Calibration uses invasive arterial pressure as a cuff surrogate in this dataset.
