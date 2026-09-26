# Tiny BP for ESP32-S3

This directory holds a PPG-only 1D CNN experiment using the eligible VitalDB cases and its ESP32-S3 firmware. It is separate from the repository's PulseDB experiments. Training data are not committed here.

## Model and firmware files

| Path | Purpose |
|---|---|
| `model/best.pt` | PyTorch weights (11,818 parameters) |
| `model/tiny_bp.onnx` | Exported float model |
| `model/preprocess.json` | 125 Hz, 1,250 samples, training mean/std, output scaling |
| `model/metrics.json` | Subject-disjoint validation and test results |
| `model/tiny_bp_kl_eval.json` | ESP-PPQ INT8 simulation on test windows |
| `model/split_subjects.json` | Subject IDs for the fixed train/validation/test split |
| `firmware/main/models/s3/model.espdl` | ESP32-S3 INT8 model used by firmware |
| `comparison/same_windows_metrics.json` | Tiny BP and retrained ResNet metrics on identical VitalDB test windows |
| `ablation/` | Matched PPG upstroke experiment, four weights, and test metrics |
| `ecg_ppg/` | Matched VitalDB ECG + PPG training, TensorBoard runs, weights, and paired evaluation |

The training pool contained 3,075 subjects and 307,079 windows; validation had 26 subjects / 2,600 windows; the held-out test had 32 subjects / 3,200 windows. On that test set, the float model's SBP/DBP MAE was **13.47 / 7.26 mmHg**. The ESP-PPQ simulated INT8 MAE was **13.62 / 7.25 mmHg**. These are VitalDB operating-room PPG results, not wrist or clinical validation. The test set was inspected during model development, so model-selection claims need a new independent test set.

The [upstroke experiment](ablation/README.md) retrains this architecture with altered PPG rising edges and a matched raw-waveform control. Its 20-epoch raw control is separate from the longer-trained model above.

The [ECG + PPG experiment](ecg_ppg/README.md) recovers synchronized lead-II ECG for 2,955 of the same 3,200 test windows, then compares PPG-only, early fusion, and frozen-PPG ECG correction on identical retained windows. Across three seeds, the ECG correction produced a small, uncertain SBP MAE reduction and a small DBP MAE increase. Explicit ECG-to-PPG timing features did not show a reliable additional benefit.

## Comparison on the same test windows

`compare_on_vitaldb.py` trains the repository's ResNet-1D or ResNet-BiGRU architecture on the same 307,079 training windows and selects each target's best checkpoint on the same 2,600 validation windows. The two targets are trained separately, following the original architecture's scalar-output design. `evaluate_on_vitaldb.py` evaluates those weights and Tiny BP with FP32 inference on the same 3,200 test windows and bootstraps the paired MAE difference by case. Results are in `comparison/same_windows_metrics.json`; the four retrained weights are in `comparison/weights/`. The models share data and preprocessing, while Tiny BP has a different training schedule. These weights were trained for this comparison and are not the original project's PulseDB weights.

```bash
python tiny-bp-esp32/compare_on_vitaldb.py --train path/to/Train_Subset.npz --test path/to/CalFree_Test_Subset.npz --split tiny-bp-esp32/model/split_subjects.json --architecture resnet1d --epochs 12 --out comparison_run
python tiny-bp-esp32/compare_on_vitaldb.py --train path/to/Train_Subset.npz --test path/to/CalFree_Test_Subset.npz --split tiny-bp-esp32/model/split_subjects.json --architecture resnet_bigru --epochs 12 --out comparison_run
python tiny-bp-esp32/evaluate_on_vitaldb.py --test path/to/CalFree_Test_Subset.npz --out comparison_run/same_windows_metrics.json --weights comparison_run
```

`model.py`, `train.py`, `export_trained.py`, and `quantize.py` contain the model and training/export code. `train.py` accepts NPZ arrays `ppg_signals`, `sbp`, `dbp`, and `subjects`; the full raw dataset is external and is not in Git. The included weights came from a CPU training stage followed by GPU continuation from the best CPU weights. See the script arguments for starting a new run. Install training and quantization requirements in separate Python environments because their ONNX dependencies differ.

## Firmware and sample input

The ESP-IDF project is in `firmware/`. Its default source replays one **real** 10-second VitalDB training PPG window, case 2158 at 1856 seconds, through `bp::SampleSource`. The reference pressure is 119.90/79.99 mmHg, while the float model produces 106.76/60.59 mmHg for this example. This is an input/inference fixture and visibly illustrates model error. Its values and provenance are in `firmware/main/ppg_fixture.hpp` and `.json`.

The shared sample contract and instructions for replacing the fixture with a 125 Hz live sensor callback are in [firmware/SENSOR_INTERFACE.md](firmware/SENSOR_INTERFACE.md). A window contains 10 seconds of raw PPG; after the first window, another can be inferred every second. Any lost sample resets the window.

With ESP-IDF installed:

```bash
cd tiny-bp-esp32/firmware
idf.py set-target esp32s3
idf.py build
idf.py -p YOUR_PORT flash monitor
```

This project has not yet been compiled or flashed on a XIAO ESP32-S3 because that toolchain and board were unavailable. The embedded waveform is finger PPG; a wrist sensor needs signal-scale alignment, synchronized wrist reference data, and its own evaluation before interpreting BP output.
