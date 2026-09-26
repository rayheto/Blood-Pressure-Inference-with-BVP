"""Replay held-out cases chronologically with one initial BP calibration per case."""

import argparse
import json
from pathlib import Path

import numpy as np

from stream_inference import StreamingBloodPressure
from train_fusion import load_matched
from train_delta import score


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--test", type=Path, required=True)
    p.add_argument("--ecg-dir", type=Path, required=True)
    p.add_argument("--preprocess", type=Path, required=True)
    p.add_argument("--run", type=Path, required=True)
    args = p.parse_args()
    ppg, ecg, y, _, cases, times, _ = load_matched(args.test, args.ecg_dir / "test")
    models = {name: StreamingBloodPressure(args.run / f"{name}.pt", args.preprocess, mode=name)
              for name in ("ppg_only", "ppg_pat_rr")}
    actual, predicted, baseline, pair_cases, ages = [], {name: [] for name in models}, [], [], []
    for case in np.unique(cases):
        ids = np.flatnonzero(cases == case)
        ids = ids[np.argsort(times[ids])]
        first = int(ids[0])
        for model in models.values():
            model.calibrate(ppg[first], ecg[first], float(y[first, 0]), float(y[first, 1]), float(times[first]))
        for later in ids[1:]:
            age = float(times[later] - times[first])
            if age < 10 or age > 1800:
                continue
            outputs = {name: model.observe_window(ppg[later], ecg[later], float(times[later]))
                       for name, model in models.items()}
            if any(value["status"] != "estimate" for value in outputs.values()):
                raise RuntimeError(f"No estimate for case {case}, age {age}")
            actual.append(y[later] - y[first])
            baseline.append([0., 0.])
            for name, value in outputs.items():
                predicted[name].append([value["delta_sbp_mmhg"], value["delta_dbp_mmhg"]])
            pair_cases.append(case)
            ages.append(age)
    truth = np.asarray(actual)
    pred = {key: np.asarray(value) for key, value in predicted.items()}
    pair_cases = np.asarray(pair_cases)
    ages = np.asarray(ages)
    report = {"cases": int(len(np.unique(pair_cases))), "evaluated_windows": len(truth),
              "calibration": "first available test window in each case; invasive ART used to emulate cuff",
              "horizon_seconds": 1800,
              "zero_change": score(np.zeros_like(truth), truth),
              "models": {key: score(value, truth) for key, value in pred.items()},
              "by_elapsed_time": {}}
    for low, high in ((10, 300), (300, 900), (900, 1801)):
        mask = (ages >= low) & (ages < high)
        if np.any(mask):
            report["by_elapsed_time"][f"{low}-{high-1}s"] = {
                "windows": int(mask.sum()),
                "models": {key: score(value[mask], truth[mask]) for key, value in pred.items()}}
    groups = [np.flatnonzero(pair_cases == case) for case in np.unique(pair_cases)]
    rng = np.random.default_rng(2026)
    boot = []
    for _ in range(2000):
        ids = np.concatenate([groups[i] for i in rng.integers(len(groups), size=len(groups))])
        boot.append((np.abs(pred["ppg_pat_rr"][ids] - truth[ids]).mean(axis=0) -
                     np.abs(pred["ppg_only"][ids] - truth[ids]).mean(axis=0)))
    report["pat_rr_minus_ppg_mae_95pct_sbp_dbp"] = np.quantile(boot, [.025, .975], axis=0).T.tolist()
    (args.run / "stream_replay_metrics.json").write_text(json.dumps(report, indent=2))
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
