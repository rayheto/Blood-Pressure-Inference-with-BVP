"""Paired case-level comparison of models on synchronized held-out windows."""

import argparse
import json
from pathlib import Path

import numpy as np

from train_fusion import load_matched


def trend_pairs(cases, times):
    pairs = []
    for case in np.unique(cases):
        ids = np.flatnonzero(cases == case)
        ids = ids[np.argsort(times[ids])]
        for pos, i in enumerate(ids[:-1]):
            later = ids[pos + 1:]
            eligible = later[(times[later] - times[i] >= 60) &
                             (times[later] - times[i] <= 300)]
            if len(eligible):
                pairs.append((i, eligible[0]))
    return np.asarray(pairs, dtype=np.int64).reshape(-1, 2)


def trend(pred, actual, pairs):
    if not len(pairs):
        return None
    a, b = pairs.T
    truth = actual[b] - actual[a]
    estimate = pred[b] - pred[a]
    return {"pairs": len(pairs),
            "sbp_change_mae": float(np.abs(estimate[:, 0] - truth[:, 0]).mean()),
            "dbp_change_mae": float(np.abs(estimate[:, 1] - truth[:, 1]).mean()),
            "sbp_change_correlation": float(np.corrcoef(estimate[:, 0], truth[:, 0])[0, 1]),
            "dbp_change_correlation": float(np.corrcoef(estimate[:, 1], truth[:, 1])[0, 1])}


def trend_difference_bootstrap(pred, baseline, actual, cases, pairs):
    if not len(pairs):
        return None
    a, b = pairs.T
    pair_cases = cases[a]
    case_ids = np.unique(pair_cases)
    groups = [np.flatnonzero(pair_cases == case) for case in case_ids]
    truth = actual[b] - actual[a]
    candidate = pred[b] - pred[a]
    control = baseline[b] - baseline[a]
    rng = np.random.default_rng(42)
    corr_diff = []
    change_mae_diff = []
    for _ in range(2000):
        chosen = rng.integers(len(groups), size=len(groups))
        idx = np.concatenate([groups[i] for i in chosen])
        corr_diff.append([np.corrcoef(candidate[idx, k], truth[idx, k])[0, 1] -
                          np.corrcoef(control[idx, k], truth[idx, k])[0, 1]
                          for k in (0, 1)])
        change_mae_diff.append((np.abs(candidate[idx] - truth[idx]).mean(axis=0) -
                                np.abs(control[idx] - truth[idx]).mean(axis=0)).tolist())
    return {"correlation_difference_95pct_sbp_dbp":
            np.quantile(corr_diff, [0.025, 0.975], axis=0).T.tolist(),
            "change_mae_difference_95pct_sbp_dbp":
            np.quantile(change_mae_diff, [0.025, 0.975], axis=0).T.tolist()}


def case_bootstrap(pred, baseline, actual, cases):
    per_case = np.stack([
        np.abs(pred[cases == c] - actual[cases == c]).mean(axis=0) -
        np.abs(baseline[cases == c] - actual[cases == c]).mean(axis=0)
        for c in np.unique(cases)
    ])
    rng = np.random.default_rng(42)
    replicates = per_case[rng.integers(len(per_case), size=(5000, len(per_case)))].mean(axis=1)
    return {"mean_case_sbp_mae_difference": float(per_case[:, 0].mean()),
            "mean_case_dbp_mae_difference": float(per_case[:, 1].mean()),
            "sbp_95pct": np.quantile(replicates[:, 0], [0.025, 0.975]).tolist(),
            "dbp_95pct": np.quantile(replicates[:, 1], [0.025, 0.975]).tolist()}


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--test", type=Path, required=True)
    p.add_argument("--ecg-dir", type=Path, required=True)
    p.add_argument("--run", type=Path, required=True)
    args = p.parse_args()
    _, _, actual, _, cases, times, corr = load_matched(args.test, args.ecg_dir / "test")
    trained = json.loads((args.run / "metrics.json").read_text())["models"]
    predictions = {mode: np.load(args.run / f"{mode}_test_predictions.npy")
                   for mode in trained}
    pairs = trend_pairs(cases, times)
    report = {"windows": len(actual), "cases": len(np.unique(cases)),
              "minimum_ppg_match_correlation": float(corr.min()), "modes": {}}
    for mode, pred in predictions.items():
        if pred.shape != actual.shape:
            raise ValueError(f"{mode}: prediction and target shape mismatch")
        error = pred - actual
        report["modes"][mode] = {
            "sbp_mae": float(np.abs(error[:, 0]).mean()),
            "dbp_mae": float(np.abs(error[:, 1]).mean()),
            "sbp_error_sd": float(error[:, 0].std(ddof=1)),
            "dbp_error_sd": float(error[:, 1].std(ddof=1)),
            "within_case_60_300s": trend(pred, actual, pairs),
        }
        if mode != "ppg_only":
            report["modes"][mode]["vs_ppg_only"] = case_bootstrap(
                pred, predictions["ppg_only"], actual, cases)
            report["modes"][mode]["within_case_vs_ppg_only"] = trend_difference_bootstrap(
                pred, predictions["ppg_only"], actual, cases, pairs)
    out = args.run / "paired_metrics.json"
    out.write_text(json.dumps(report, indent=2))
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
