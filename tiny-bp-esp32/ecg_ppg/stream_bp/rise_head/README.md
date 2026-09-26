# Rise-aware supervised correction experiment

This is an exploratory follow-up to the [five-input dynamic head](../dynamic_head/README.md). The original `ppg_pat_rr.pt` encoder and change predictor remain frozen. A small bounded residual head receives both frozen 32-channel waveform embeddings, the original BP-change prediction, calibration SBP/DBP, elapsed time, and PAT/RR timing features. The adjustment is limited to ±80 mmHg SBP and ±40 mmHg DBP. No current or future arterial-pressure value is an inference input.

Training used the same 51,991 first-calibration VitalDB pairs, with 446 validation pairs and 784 held-out test pairs. It sampled training pairs across four SBP-change bins and cases, and mildly penalized underestimation for true rises of at least 20 mmHg. The best of 100 possible epochs was selected at epoch 22 using validation overall SBP/DBP MAE plus 0.35 times the SBP MAE for rises of at least 40 mmHg. TensorBoard run: `VitalDB_stream_rise_head` under `D:/BP_Training/all_eligible/tensorboard_merged`.

## Numeric results

The same held-out cases and windows are used for all models. MAE is in mmHg and scored on **unsmoothed** windows.

| Dataset / model | SBP change MAE | DBP change MAE | SBP MAE when rise ≥40 |
|---|---:|---:|---:|
| Validation: frozen base | 18.25 | 8.72 | 43.17 (70 windows) |
| Validation: rise head | **12.74** | **5.83** | **24.06** (70 windows) |
| Held out: frozen base | 13.71 | **6.26** | 10.66 (38 windows) |
| Held out: prior uniform head | **10.41** | 6.28 | 8.30 (38 windows) |
| Held out: rise head | 11.00 | 6.54 | **7.92** (38 windows) |

The 38 held-out rise≥40 windows come from only **eight cases**. A case-bootstrap 95% interval for the rise head's overall SBP MAE minus the frozen base is **[-5.99, +1.25] mmHg**; versus the prior uniform head it is **[-1.56, +2.72]**. Both include zero. For rise≥40 SBP, the corresponding intervals are **[-15.58, +3.93]** versus base and **[-8.75, +4.21]** versus uniform. This test does not establish a reliable improvement.

One original continuous recording, held-out VitalDB case 236, was replayed for 175 valid 10-second windows after simulated calibration:

| Model | SBP change MAE | DBP change MAE |
|---|---:|---:|
| Frozen base | 26.51 | 13.66 |
| Prior uniform head | 14.97 | 9.71 |
| Rise head | **8.07** | **5.46** |

For this case's 15 windows at reference SBP 160–180 mmHg, SBP MAE was **55.89** base, **50.04** prior uniform, and **26.98** rise head. Its curve was inspected before designing this experiment. It therefore cannot serve as a fresh independent confirmation, even though its recording belongs to the held-out split. Repeated inspection of the same held-out set also limits model ranking. Testing on entirely new cases with wrist PPG and external cuff measurements is still required.

The [animated SVG](case_236_rise_fit.svg) shows **four curves only**: arterial-pressure reference, frozen base, prior uniform head, and rise head. All are displayed with the same trailing 60-second average; the numeric results above use unsmoothed values. `case_236_trace.json` contains every plotted estimate. The animation can be regenerated with `python create_animation.py`.

## Artifacts

- `rise_head.pt`: PyTorch research checkpoint.
- `metrics.json`: validation and held-out scores.
- `test_case_bootstrap.json`: paired case-bootstrap differences.
- `raw_stream_case_236.json` and `case_236_trace.json`: continuous replay metrics and sequence.
- `../../train_rise_head.py`, `../../evaluate_rise_head.py`: training and uncertainty analysis.
- `../../stream_inference.py`: pass `rise_checkpoint=Path('.../rise_head.pt')` with `mode='ppg_pat_rr'` to run this head.

The model is not quantized or deployed on ESP32-S3. Arterial pressure simulates a one-time cuff calibration; this remains an operating-room data experiment.
