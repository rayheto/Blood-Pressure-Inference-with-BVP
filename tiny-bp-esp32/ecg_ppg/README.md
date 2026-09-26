# Synchronized ECG + PPG training

The [streaming BP change experiment](stream_bp/README.md) extends this work to cuff-calibrated, 10-second online PPG + ECG windows with a 30-minute horizon. It includes final checkpoints, chronological replay, and a continuous raw-recording smoke test.

The separate [within-case 60–300-second BP change pilot](delta_60_300s/README.md) directly trains paired-window change predictors with PPG alone and with ECG + PPG. Its checkpoints, test metrics, and TensorBoard labels are documented there.
The [explicit ECG timing follow-up](delta_pat_60_300s/README.md) compares the fixed PPG baseline against PPG plus change in R-to-PPG-valley delay, with and without ECG R-to-R interval change, on the same window pairs.
The [five-seed repeat](delta_seed_repeats/README.md) retrains all three change predictors with seeds 42–46 and reports paired case-and-seed uncertainty.
The [ECG-only five-seed baseline](ecg_only_seed_repeats/README.md) uses the same paired windows and shows how much change information lead-II ECG provides on its own.

The existing VitalDB NPZ windows contain PPG and arterial-pressure labels, but no ECG. `build_ecg_windows.py` downloads the corresponding public `SNUADC/ECG_II` and `SNUADC/PLETH` tracks and recovers each saved PPG window's actual sample offset by waveform matching. The NPZ `time_s` field is integer seconds and is insufficient for ECG-to-PPG timing features. A window is retained only when its PPG match correlation is at least 0.96 and the aligned ECG has at least 98% finite samples with nontrivial amplitude. This produces a matched ECG subset; all compared models train and test on exactly these windows.

`train_fusion.py` trains three from-scratch models and two follow-up correction models. All use the same fixed subject split, optimizer, seed, 30-epoch maximum, and validation-checkpoint rule:

| TensorBoard run suffix | Inputs |
|---|---|
| `ppg_only` | PPG waveform |
| `ecg_ppg` | Synchronized PPG and lead-II ECG waveforms |
| `ecg_ppg_pat` | The two waveforms, plus median R-to-PPG-foot delay, ECG RR interval, and valid delay-pair count |
| `ecg_residual` | Frozen PPG baseline plus a trainable ECG correction branch, initialized to zero |
| `ecg_delay_residual` | The same correction branch with the three explicit timing features |

The R-to-foot delay is a **proxy**, not validated pulse transit time. Monitor timing, morphology, peak detection, and physiological variation may affect it. The model is a small 1D CNN comparison; its dual-channel input does not imply current firmware compatibility. Saved ECG windows and training NPZ files stay outside Git.

The PPG baseline has 11,818 parameters, the early-fusion model has 11,890, and the ECG correction model has 15,460 in total (3,642 trained in its second stage). These weights are PyTorch research artifacts; the ECG variants have not been quantized or run on the ESP32-S3.

The correction models were added after inspecting the first three models' test results. They are exploratory follow-ups, not a prespecified independent test. The same test set was also inspected in earlier PPG work, so all ranking claims need replication on new subjects and preferably another sensor setting.

Each run writes `loss/train_mse_scaled`, `loss/val_mse_scaled`, `mae/val_sbp_mmhg`, `mae/val_dbp_mmhg`, the learning rate, and final test metrics to TensorBoard. The run name and each model name appear separately. The current TensorBoard server reads `D:/BP_Training/all_eligible/tensorboard_merged` at `http://127.0.0.1:6006/`.

Example commands from the repository root:

```bash
python tiny-bp-esp32/ecg_ppg/build_ecg_windows.py --train D:/BP_Training/all_eligible/data/Train_Subset.npz --test D:/BP_Training/all_eligible/data/CalFree_Test_Subset.npz --out D:/BP_Training/ecg_ppg --workers 8
python tiny-bp-esp32/ecg_ppg/train_fusion.py --train D:/BP_Training/all_eligible/data/Train_Subset.npz --test D:/BP_Training/all_eligible/data/CalFree_Test_Subset.npz --ecg-dir D:/BP_Training/ecg_ppg --split tiny-bp-esp32/model/split_subjects.json --preprocess tiny-bp-esp32/model/preprocess.json --tensorboard D:/BP_Training/all_eligible/tensorboard_merged --run-name ECG_PPG_matched_all_cases --out D:/BP_Training/ecg_ppg/run_all_cases --epochs 30
python tiny-bp-esp32/ecg_ppg/train_fusion.py --train D:/BP_Training/all_eligible/data/Train_Subset.npz --test D:/BP_Training/all_eligible/data/CalFree_Test_Subset.npz --ecg-dir D:/BP_Training/ecg_ppg --split tiny-bp-esp32/model/split_subjects.json --preprocess tiny-bp-esp32/model/preprocess.json --tensorboard D:/BP_Training/all_eligible/tensorboard_merged --run-name ECG_PPG_matched_all_cases --out D:/BP_Training/ecg_ppg/run_all_cases --epochs 30 --modes ecg_residual ecg_delay_residual
python tiny-bp-esp32/ecg_ppg/evaluate_fusion.py --test D:/BP_Training/all_eligible/data/CalFree_Test_Subset.npz --ecg-dir D:/BP_Training/ecg_ppg --run D:/BP_Training/ecg_ppg/run_all_cases
```

To repeat the correction comparison, run `train_fusion.py` with `--seed 43` or `--seed 44`, a separate `--out` directory and `--run-name`, and `--modes ppg_only ecg_residual ecg_delay_residual`. Then run `evaluate_fusion.py` for each output. `collate_results.py` combines the three paired reports and copies their best weights into this directory.

The paired evaluation reports absolute error and 60–300-second within-case BP change metrics, with a case-level bootstrap interval for the MAE difference against PPG alone. These are operating-room signals and invasive arterial-pressure labels; wrist-sensor and clinical performance remain untested.

## Matched VitalDB results

The acquisition retained 276,819 training, 2,422 validation, and 2,955 test windows. The split contains 3,028 training, 26 validation, and 32 test subjects. The table below reports seed 42's best validation checkpoint on the identical 2,955 test windows. MAE is in mmHg; change correlation uses 2,600 within-case window pairs separated by 60–300 seconds.

| Input / fusion | SBP MAE | DBP MAE | SBP change correlation | DBP change correlation |
|---|---:|---:|---:|---:|
| PPG only | 13.49 | 7.43 | 0.509 | 0.485 |
| ECG + PPG, early fusion | 13.94 | 7.52 | 0.462 | 0.448 |
| ECG + PPG + timing features, early fusion | 13.97 | 7.75 | 0.472 | 0.439 |
| Frozen PPG + ECG correction | 13.28 | 7.57 | 0.530 | 0.490 |
| Frozen PPG + ECG correction + timing features | 13.51 | 7.55 | 0.536 | 0.495 |

For the frozen-PPG ECG correction, seed 42's case-paired bootstrap 95% intervals for MAE minus PPG were **[-0.91, +0.54] mmHg SBP** and **[-0.32, +0.55] mmHg DBP**. Both include zero. The early-fusion models had higher short-term change MAE than PPG, and the case-bootstrap intervals for those change-error increases excluded zero in seed 42. Timing-feature gains in correlation were small and inconsistent across seed-level intervals.

The frozen-PPG correction was repeated with seeds 42, 43, and 44. Relative to each seed's own PPG baseline, its SBP MAE changed by **-0.21, -0.50, and -0.11 mmHg** (mean -0.27), while DBP MAE changed by **+0.14, +0.16, and +0.08 mmHg** (mean +0.13). Adding explicit timing features did not consistently improve these values. These are repeated optimizations on one inspected test set, not three independent populations. See [results.json](results.json) for every run, paired intervals, and the saved best [weights](weights/). TensorBoard runs are `ECG_PPG_matched_all_cases`, `ECG_PPG_seed43_matched_all_cases`, and `ECG_PPG_seed44_matched_all_cases`.

Data source: [VitalDB Open Dataset API](https://api.vitaldb.net/), [track descriptions](https://vitaldb.net/dataset/).
