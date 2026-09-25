# PPG upstroke experiment

This experiment tests whether altering pulse rising-edge shape affects Tiny BP's blood-pressure estimates. A waveform-derived feature cannot literally be deleted while leaving the waveform unchanged. Each transformation retains 1,250 samples per window, the original sample rate, detected peak positions and values, and the remaining waveform outside the altered segments.

| Training and test input | Alteration |
|---|---|
| `raw` | Original PPG waveform; newly trained matched control |
| `linear_rise` | Interpolate each detected trough-to-peak rise linearly; keeps rise duration and amplitude while removing much of its curvature |
| `fixed_rise` | Hold the trough level, then use a 20-sample ramp to the original peak; reduces variation in the final rise shape |
| `early_fall_control` | Linearly interpolate a similarly sized post-peak segment |

Each mode trained the same 11,818-parameter CNN from scratch with seed 42, Adam learning rate 0.001, batch size 256, 20 epochs, and the checkpoint with the lowest validation mean squared error. Training had 307,079 windows from 3,075 subjects; validation had 2,600 windows from 26 subjects. The four saved checkpoints were evaluated on the same 3,200 windows from 32 other subjects. In a 1,000-window transformation check, all modes preserved all 11,820 detected peaks. The altered modes changed roughly 31% of samples. These checks establish waveform length and peak preservation, not physiological equivalence.

| Input | SBP MAE | DBP MAE | SBP change correlation | DBP change correlation |
|---|---:|---:|---:|---:|
| Raw control | 13.64 | 7.54 | 0.457 | 0.455 |
| Linear rise | 13.65 | 7.51 | 0.482 | 0.460 |
| Fixed rise | 13.83 | 7.36 | 0.499 | 0.501 |
| Early-fall control | 13.87 | 8.00 | 0.391 | 0.384 |

MAE is in mmHg. Change correlations compare predicted and reference BP changes between windows from the same case separated by 60–300 seconds (2,842 pairs). For `linear_rise`, the case-paired bootstrap 95% interval of MAE minus raw control was **[-0.45, 0.46] SBP** and **[-0.50, 0.43] DBP**. For `fixed_rise`, it was **[-0.40, 0.80] SBP** and **[-0.47, 0.12] DBP**. These intervals include zero. The early-fall control's DBP difference interval was **[0.03, 0.89]**. Full-precision results are in [metrics.json](metrics.json); the best checkpoints are in [weights/](weights/).

The upstroke variants did not produce a clear loss of absolute BP accuracy. This does not establish that the network never learned rising-edge information: rise duration, amplitude, peak timing, and correlated information elsewhere in the PPG remain available. The fixed-rise transformation also retains a variable plateau duration. The change-correlation differences are exploratory and do not establish better tracking. Results use operating-room PPG; the test set was inspected during development. This is not wrist or clinical validation.

To reproduce from the repository root, provide the same VitalDB NPZ files and a Python environment with the project's training dependencies:

```bash
python tiny-bp-esp32/upstroke_ablation.py --train path/to/Train_Subset.npz --test path/to/CalFree_Test_Subset.npz --split tiny-bp-esp32/model/split_subjects.json --preprocess tiny-bp-esp32/model/preprocess.json --mode raw --epochs 20 --out ablation_run
python tiny-bp-esp32/upstroke_ablation.py --train path/to/Train_Subset.npz --test path/to/CalFree_Test_Subset.npz --split tiny-bp-esp32/model/split_subjects.json --preprocess tiny-bp-esp32/model/preprocess.json --mode linear_rise --epochs 20 --out ablation_run
python tiny-bp-esp32/upstroke_ablation.py --train path/to/Train_Subset.npz --test path/to/CalFree_Test_Subset.npz --split tiny-bp-esp32/model/split_subjects.json --preprocess tiny-bp-esp32/model/preprocess.json --mode fixed_rise --epochs 20 --out ablation_run
python tiny-bp-esp32/upstroke_ablation.py --train path/to/Train_Subset.npz --test path/to/CalFree_Test_Subset.npz --split tiny-bp-esp32/model/split_subjects.json --preprocess tiny-bp-esp32/model/preprocess.json --mode early_fall_control --epochs 20 --out ablation_run
python tiny-bp-esp32/evaluate_upstroke_ablation.py --test path/to/CalFree_Test_Subset.npz --split tiny-bp-esp32/model/split_subjects.json --preprocess tiny-bp-esp32/model/preprocess.json --weights ablation_run --out ablation_run/metrics.json
```
