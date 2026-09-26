"""Paired case-bootstrap comparison for the rise-aware correction experiment."""

import argparse
import json
from pathlib import Path

import numpy as np

from train_fusion import load_matched
from train_stream_bp import first_calibration_pairs


def main():
    p = argparse.ArgumentParser()
    for name in ("test", "ecg-dir", "base-cache", "uniform", "rise", "output"):
        p.add_argument("--" + name, type=Path, required=True)
    args = p.parse_args()
    _, _, _, _, cases, times, _ = load_matched(args.test, args.ecg_dir / "test")
    pairs = first_calibration_pairs(cases, times)
    pair_cases = cases[pairs[:, 0]]
    case_ids = np.unique(pair_cases)
    groups = [np.flatnonzero(pair_cases == case) for case in case_ids]
    with np.load(args.base_cache) as z:
        base, truth = z["pred"], z["truth"]
    preds = {"base": base, "uniform": np.load(args.uniform), "rise": np.load(args.rise)}
    mask = truth[:, 0] >= 40
    rng = np.random.default_rng(2026)
    report = {"held_out_cases": len(groups), "windows": len(truth),
              "rise_ge_40_windows": int(mask.sum()),
              "rise_ge_40_cases": int(np.unique(pair_cases[mask]).size),
              "case_bootstrap_samples": 2000, "comparisons": {}}
    for subset, keep in (("all", np.ones(len(truth), bool)), ("rise_ge_40", mask)):
        for against in ("base", "uniform"):
            key = f"rise_minus_{against}_{subset}"
            difference = np.abs(preds["rise"] - truth) - np.abs(preds[against] - truth)
            point = difference[keep].mean(axis=0)
            boot = []
            for _ in range(2000):
                ids = np.concatenate([groups[j] for j in rng.integers(len(groups), size=len(groups))])
                ids = ids[keep[ids]]
                if len(ids):
                    boot.append(difference[ids].mean(axis=0))
            report["comparisons"][key] = {
                "mae_difference_sbp_dbp": point.tolist(),
                "case_bootstrap_95pct_sbp_dbp": np.quantile(boot, [.025, .975], axis=0).T.tolist(),
            }
    args.output.write_text(json.dumps(report, indent=2))
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
