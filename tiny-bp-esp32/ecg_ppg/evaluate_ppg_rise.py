"""Paired case-bootstrap comparison of PPG-only and ECG-assisted rise heads."""

import argparse
import json
from pathlib import Path

import numpy as np

from train_fusion import load_matched
from train_stream_bp import first_calibration_pairs


def main():
    p = argparse.ArgumentParser()
    for name in ("test", "ecg-dir", "ppg", "fusion", "output"):
        p.add_argument("--" + name, type=Path, required=True)
    args = p.parse_args()
    _, _, _, _, cases, times, _ = load_matched(args.test, args.ecg_dir / "test")
    pairs = first_calibration_pairs(cases, times)
    case_for_pair = cases[pairs[:, 0]]
    groups = [np.flatnonzero(case_for_pair == case) for case in np.unique(case_for_pair)]
    with np.load(args.ppg / "test_state_cache.npz") as z:
        truth = z["truth"]
        ppg_base = z["base"]
    with np.load(args.fusion / "test_state_cache.npz") as z:
        if not np.allclose(truth, z["truth"]):
            raise ValueError("PPG and fusion label sequences differ")
        fusion_base = z["base"]
    predictions = {
        "ppg_base": ppg_base,
        "ppg_rise": np.load(args.ppg / "rise_head_test_predictions.npy"),
        "fusion_base": fusion_base,
        "fusion_rise": np.load(args.fusion / "rise_head_test_predictions.npy"),
    }
    mask = truth[:, 0] >= 40
    report = {"cases": len(groups), "windows": len(truth), "rise_ge_40_windows": int(mask.sum()),
              "rise_ge_40_cases": int(np.unique(case_for_pair[mask]).size), "models": {}, "comparisons": {}}
    for name, pred in predictions.items():
        report["models"][name] = {
            "sbp_mae": float(np.abs(pred[:, 0] - truth[:, 0]).mean()),
            "dbp_mae": float(np.abs(pred[:, 1] - truth[:, 1]).mean()),
            "rise_ge_40_sbp_mae": float(np.abs(pred[mask, 0] - truth[mask, 0]).mean()),
        }
    rng = np.random.default_rng(2026)
    for subset, keep in (("all", np.ones(len(truth), bool)), ("rise_ge_40", mask)):
        for name_a, name_b in (("ppg_rise", "fusion_rise"), ("ppg_rise", "ppg_base")):
            difference = np.abs(predictions[name_a] - truth) - np.abs(predictions[name_b] - truth)
            boot = []
            for _ in range(2000):
                indices = np.concatenate([groups[j] for j in rng.integers(len(groups), size=len(groups))])
                indices = indices[keep[indices]]
                if len(indices):
                    boot.append(difference[indices].mean(axis=0))
            report["comparisons"][name_a + "_minus_" + name_b + "_" + subset] = {
                "mae_difference_sbp_dbp": difference[keep].mean(axis=0).tolist(),
                "case_bootstrap_95pct_sbp_dbp": np.quantile(boot, [.025, .975], axis=0).T.tolist(),
            }
    args.output.write_text(json.dumps(report, indent=2))
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
