# Current-PPG base model

This is the selected **current-window PPG-only** absolute BP model. It combines the tracked TinyBP CNN in `../model/best.pt` with the selected correction head in `selected_head.npz`. It does not use ECG, a personal cuff value, or prior windows. The head was selected among three seeds using validation pressure-weighted SBP+DBP MAE plus 0.25 times the 60-second dynamic error. Seed **20260929** had the lowest validation score.

The input is a chronological 10-second raw PPG window, 1,250 samples at 125 Hz, on the VitalDB signal scale. The output is SBP and DBP in mmHg. `export_firmware.py` fuses convolution BatchNorm parameters and writes the FP32 arrays used in `../firmware/main/ppg_base_weights.hpp`; it also refreshes the real-waveform fixture expectation. The `.npz` contains only the small correction head and feature normalization, so the firmware export is reproducible from tracked artifacts.

## Evaluation boundary

On five reused, case-disjoint VitalDB continuous test cases (4,198 valid 10-second windows), this seed had training-distribution-weighted SBP/DBP MAE **12.672/6.981 mmHg**. For reference, the uncorrected TinyBP on the same windows had **17.502/7.869 mmHg**. Full bin counts, per-bin MAE, 60/300-second change errors, and direction accuracy are in [metrics.json](metrics.json). The test cases were inspected during research; these numbers are not a fresh blind estimate. VitalDB operating-room finger PPG is not wrist PPG.

The firmware runs this model in FP32 with static scratch arrays. No ESP-DL quantization or ESP32-S3 timing claim is transferred from the previous TinyBP INT8 firmware. The first hardware check should compare its real-fixture output with the PC value in `../firmware/main/ppg_fixture.json`, then time inference on the board. The fixture has a real PPG waveform and reference pressure, but it is a functional check, not an accuracy test.

To regenerate the firmware arrays from tracked files:

```bash
python tiny-bp-esp32/ppg_base/export_firmware.py
```
