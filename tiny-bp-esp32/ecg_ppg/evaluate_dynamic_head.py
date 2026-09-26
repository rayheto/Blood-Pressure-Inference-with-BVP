"""Case-bootstrap uncertainty for the held-out dynamic-head comparison."""

import argparse
import json
from pathlib import Path

import numpy as np

from train_fusion import load_matched
from train_stream_bp import first_calibration_pairs


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--test", type=Path, required=True)
    parser.add_argument("--ecg-dir", type=Path, required=True)
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    _, _, _, _, cases, times, _ = load_matched(args.test, args.ecg_dir / "test")
    pairs = first_calibration_pairs(cases, times)
    groups = [np.flatnonzero(cases[pairs[:, 0]] == case)
              for case in np.unique(cases[pairs[:, 0]])]
    with np.load(args.run / "test_base_cache.npz") as data:
        base, truth = data["pred"], data["truth"]
    rng = np.random.default_rng(2026)
    report = {"cases": len(groups), "windows": len(pairs), "case_bootstrap_samples": 2000, "modes": {}}
    for name in ("uniform", "high_pressure_weighted"):
        pred = np.load(args.run / f"{name}_test_predictions.npy")
        difference = np.abs(pred - truth) - np.abs(base - truth)
        point = difference.mean(axis=0)
        boot = np.empty((2000, 2), np.float64)
        for i in range(2000):
            idx = np.concatenate([groups[j] for j in rng.integers(len(groups), size=len(groups))])
            boot[i] = difference[idx].mean(axis=0)
        report["modes"][name] = {
            "mae_difference_vs_base_sbp_dbp": point.tolist(),
            "case_bootstrap_95pct_sbp_dbp": np.quantile(boot, [.025, .975], axis=0).T.tolist(),
        }
    args.output.write_text(json.dumps(report, indent=2))
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
