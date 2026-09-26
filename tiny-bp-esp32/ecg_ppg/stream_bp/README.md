# Streaming blood pressure change experiment

The [supervised dynamic correction experiment](dynamic_head/README.md) adds two bounded correction heads, reports held-out and continuous-case metrics, and includes an updated animated SVG comparison.

This research prototype accepts synchronized 125 Hz PPG and ECG in completed 10-second windows. One externally measured cuff SBP/DBP value initializes a session. Each later window produces an estimate of change from that calibration, plus the calibrated BP. It requires recalibration after 30 minutes. `stream_inference.py` also accepts smaller ordered chunks through `push_samples` and buffers them until a complete window arrives. The current implementation uses PyTorch on a computer; it has **not** been converted to or tested on ESP32-S3.

## Data and training

The experiment used all **locally available matched VitalDB** PPG, lead-II ECG, and arterial-pressure windows. The initial chronological-pair stage had 1,182,458 eligible pairs across 3,027 training cases, with one future partner sampled per anchor per epoch. Its checkpoints initialized the final first-calibration stage, which trained on 51,991 pairs across 3,021 cases. Windows are 1,250 samples at 125 Hz. The model takes calibration PPG, current PPG, and elapsed time. The fusion version also takes changes in ECG R-to-PPG-valley delay (an approximate PAT feature), ECG RR interval, and timing-validity flags. This delay has not been validated as physical pulse transit time.

Only the final adapted checkpoints are included here. The full PulseDB archive was not locally available and was **not** included in this run. The data comes from operating-room instruments; there is no wrist-PPG or real cuff validation.

## Held-out evaluation

For each of 32 held-out VitalDB cases, the first available window supplied a simulated calibration value from invasive arterial pressure. `evaluate_stream_replay.py` then fed later windows in time order to the online inference interface. There were 784 evaluated windows within 30 minutes. Those stored windows are **sparse**, so this is not a population-level continuous-stream validation.

| Model | SBP change MAE | DBP change MAE | SBP change correlation | DBP change correlation |
|---|---:|---:|---:|---:|
| No change from calibration | 25.08 | 11.05 | 0 | 0 |
| PPG only | 14.26 | 6.13 | 0.829 | 0.826 |
| PPG + PAT proxy change + RR change | 13.71 | 6.26 | 0.843 | 0.817 |

MAE is in mmHg; lower is better. Case-bootstrap 95% intervals for fusion minus PPG MAE are **[-1.70, +0.80] SBP** and **[-0.67, +0.75] DBP**. Both include zero, so this experiment does not establish an ECG benefit.

A separate engineering replay used one original, continuous 500 Hz VitalDB recording (case 236), downsampled to 125 Hz. It yielded 175 valid 10-second windows after a simulated calibration. Fusion MAE was **26.51 SBP / 13.66 DBP**; PPG-only was **29.29 / 15.05**, while keeping calibration unchanged was **17.95 / 9.12**. The model therefore performed worse than the simple no-change reference on this case. This one-case result is a warning about real-stream behavior, not a population estimate. See `raw_stream_case_236.json`.

[Animated case-236 fit](case_236_stream_fit.svg) shows the reference and both model outputs over the 30-minute replay. The underlying 175 timestamped values are in `case_236_trace.json`; `create_animation.py` regenerates the SVG. The animation reveals that predictions follow the direction of the rise but substantially understate its magnitude.

The [smoothed animation](case_236_stream_fit_smoothed.svg) applies the same trailing 60-second average to all three traces for viewing. This does not change the model or the original MAE reported above. Regenerate it with `python create_animation.py --smooth-seconds 60`.

The arterial-pressure waveform supplied evaluation labels and the simulated calibration reading only; it is not an inference input after calibration. No result here supports clinical use or an AAMI claim.

## Files and reproduction

- `ppg_pat_rr.pt`, `ppg_only.pt`: adapted PyTorch checkpoints.
- `metrics.json`: first-calibration training and held-out pair metrics.
- `stream_replay_metrics.json`: chronological replay metrics and elapsed-time groups.
- `raw_stream_case_236.json`: original continuous recording smoke test.
- `../stream_inference.py`: `StreamingBloodPressure` interface.
- `../train_stream_bp.py`, `../evaluate_stream_replay.py`, `../evaluate_raw_stream_case.py`: training and evaluations.
- `../../model/preprocess.json`: PPG normalization used by the runtime.

Example from the repository root, with PyTorch, NumPy, and SciPy installed:

```python
from pathlib import Path
from stream_inference import StreamingBloodPressure

model = StreamingBloodPressure(
    Path('tiny-bp-esp32/ecg_ppg/stream_bp/ppg_pat_rr.pt'),
    Path('tiny-bp-esp32/model/preprocess.json'),
)
model.calibrate(calibration_ppg_1250, calibration_ecg_1250, cuff_sbp, cuff_dbp, window_end_s=10)
result = model.observe_window(current_ppg_1250, current_ecg_1250, window_end_s=20)
```

Add `tiny-bp-esp32/ecg_ppg` to `PYTHONPATH` for this import. Both waveforms must share the same clock and sample boundaries.

TensorBoard runs are `VitalDB_stream_30min_all_windows` and `VitalDB_stream_first_calibration` under `D:/BP_Training/all_eligible/tensorboard_merged` (`http://127.0.0.1:6006/` on this machine).
