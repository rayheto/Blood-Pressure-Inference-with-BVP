"""Evaluate the saved waveform ablation models on identical held-out windows."""

import argparse
import json
from pathlib import Path

import numpy as np
import torch

from model import TinyBP
from upstroke_ablation import metrics, transform_all

MODES = ("raw", "linear_rise", "fixed_rise", "early_fall_control")


def predict(model, x, device):
    model.eval().to(device)
    with torch.no_grad():
        return np.concatenate([model(batch.to(device)).cpu().numpy()
                               for batch in x.split(256)])


def case_bootstrap(pred, reference, target, cases):
    # Difference of per-case MAEs, ablated minus raw.
    ids = np.unique(cases)
    difference = np.stack([
        np.abs(pred[cases == case] - target[cases == case]).mean(axis=0) -
        np.abs(reference[cases == case] - target[cases == case]).mean(axis=0)
        for case in ids
    ])
    rng = np.random.default_rng(42)
    boot = difference[rng.integers(len(ids), size=(5000, len(ids)))].mean(axis=1)
    return {
        "case_count": len(ids),
        "mean_case_mae_difference_sbp_dbp": difference.mean(axis=0).tolist(),
        "bootstrap_95pct_sbp": np.quantile(boot[:, 0], [0.025, 0.975]).tolist(),
        "bootstrap_95pct_dbp": np.quantile(boot[:, 1], [0.025, 0.975]).tolist(),
    }


def trend(pred, target, cases, times):
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
    i, j = np.asarray(pairs).T
    actual = target[j] - target[i]
    estimated = pred[j] - pred[i]
    return {
        "pairs": len(pairs),
        "sbp_delta_mae": float(np.abs(estimated[:, 0] - actual[:, 0]).mean()),
        "dbp_delta_mae": float(np.abs(estimated[:, 1] - actual[:, 1]).mean()),
        "sbp_delta_correlation": float(np.corrcoef(estimated[:, 0], actual[:, 0])[0, 1]),
        "dbp_delta_correlation": float(np.corrcoef(estimated[:, 1], actual[:, 1])[0, 1]),
    }


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--test", type=Path, required=True)
    p.add_argument("--preprocess", type=Path, required=True)
    p.add_argument("--split", type=Path, required=True)
    p.add_argument("--weights", type=Path, required=True)
    p.add_argument("--out", type=Path, required=True)
    args = p.parse_args()
    torch.set_num_threads(4)
    config = json.loads(args.preprocess.read_text())
    split = json.loads(args.split.read_text())
    with np.load(args.test, allow_pickle=False) as z:
        raw = z["ppg_signals"].astype(np.float32)
        target = np.stack((z["sbp"], z["dbp"]), axis=1).astype(np.float32)
        cases = z["caseids"].astype(str)
        subjects = z["subjects"].astype(str)
        times = z["time_s"].astype(np.int64)
    assert len(raw) == 3200 and set(subjects) == set(split["test"])
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    predictions = {}
    report = {"windows": len(raw), "subjects": len(np.unique(subjects)), "modes": {}}
    for mode in MODES:
        transformed, changed = transform_all(raw, mode)
        x = torch.from_numpy(((transformed - config["mean"]) /
                              config["std"]).astype(np.float32)[:, None, :])
        model = TinyBP()
        model.load_state_dict(torch.load(args.weights / f"{mode}.pt",
                                         map_location="cpu", weights_only=True))
        pred = predict(model, x, device)
        predictions[mode] = pred
        report["modes"][mode] = {
            "changed_beats": changed,
            "test": metrics(pred, target),
            "within_case_60_300s": trend(pred, target, cases, times),
        }
    for mode in MODES[1:]:
        report["modes"][mode]["vs_raw"] = case_bootstrap(
            predictions[mode], predictions["raw"], target, cases)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, indent=2))
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
