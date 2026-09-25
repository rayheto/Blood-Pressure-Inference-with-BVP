"""Evaluate saved TinyBP and newly trained repository architectures on identical windows."""

import argparse
import hashlib
import json
import sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src.dl_models import ResNet1D, ResNetBiGRU  # noqa: E402
from model import TinyBP  # noqa: E402


def predict(model, x, device, scale=1.0):
    model.eval().to(device)
    chunks = []
    with torch.no_grad():
        for batch in x.split(256):
            chunks.append(model(batch.to(device)).cpu().numpy().reshape(-1) * scale)
    return np.concatenate(chunks)


def score(pred, truth):
    error = pred - truth
    return {
        "mae_sbp": float(np.abs(error[:, 0]).mean()),
        "mae_dbp": float(np.abs(error[:, 1]).mean()),
        "bias_sbp": float(error[:, 0].mean()),
        "bias_dbp": float(error[:, 1].mean()),
        "sd_sbp": float(error[:, 0].std(ddof=1)),
        "sd_dbp": float(error[:, 1].std(ddof=1)),
        "windows": len(error),
    }


def paired_case_difference(pred, reference_pred, truth, caseids):
    """MAE(model)-MAE(TinyBP), with cases as bootstrap units."""
    rng = np.random.default_rng(42)
    unique = np.unique(caseids)
    deltas = np.stack([
        (np.abs(pred[caseids == case] - truth[caseids == case]).mean(axis=0) -
         np.abs(reference_pred[caseids == case] - truth[caseids == case]).mean(axis=0))
        for case in unique
    ])
    boot = deltas[rng.integers(len(unique), size=(5000, len(unique)))].mean(axis=1)
    return {
        "case_count": len(unique),
        "mean_case_mae_difference_sbp_dbp": deltas.mean(axis=0).tolist(),
        "bootstrap_95pct_sbp": np.quantile(boot[:, 0], [0.025, 0.975]).tolist(),
        "bootstrap_95pct_dbp": np.quantile(boot[:, 1], [0.025, 0.975]).tolist(),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--test", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--preprocess", type=Path, default=Path(__file__).parent / "model/preprocess.json")
    parser.add_argument("--tiny-weights", type=Path, default=Path(__file__).parent / "model/best.pt")
    parser.add_argument("--weights", type=Path, default=Path(__file__).parent / "comparison/weights")
    parser.add_argument("--split", type=Path, default=Path(__file__).parent / "model/split_subjects.json")
    args = parser.parse_args()
    torch.set_num_threads(4)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    config = json.loads(args.preprocess.read_text())
    split = json.loads(args.split.read_text())
    with np.load(args.test, allow_pickle=False) as data:
        x = torch.from_numpy(((data["ppg_signals"] - config["mean"]) /
                              config["std"]).astype(np.float32)[:, None, :])
        truth = np.stack((data["sbp"], data["dbp"]), axis=1).astype(np.float32)
        caseids = data["caseids"].astype(str)
        subjects = data["subjects"].astype(str)
    assert len(x) == len(truth) == len(caseids) == 3200
    assert set(subjects) == set(split["test"])
    assert set(split["train"]).isdisjoint(subjects)
    assert set(split["validation"]).isdisjoint(subjects)
    tiny = TinyBP()
    tiny.load_state_dict(torch.load(args.tiny_weights, map_location="cpu", weights_only=True))
    tiny.eval().to(device)
    chunks = []
    with torch.no_grad():
        for batch in x.split(256):
            chunks.append(tiny(batch.to(device)).cpu().numpy())
    tiny_pred = np.concatenate(chunks)
    results = {"tiny_bp": score(tiny_pred, truth)}
    for name, cls in (("resnet1d", ResNet1D), ("resnet_bigru", ResNetBiGRU)):
        columns = []
        for target in ("sbp", "dbp"):
            model = cls()
            model.load_state_dict(torch.load(args.weights / f"{name}_{target}.pt",
                                             map_location="cpu", weights_only=True))
            columns.append(predict(model, x, device, scale=100))
        pred = np.stack(columns, axis=1)
        results[name] = {
            **score(pred, truth),
            "vs_tiny_bp": paired_case_difference(pred, tiny_pred, truth, caseids),
        }
    report = {
        "test_source": "VitalDB direct PLETH/ART",
        "same_test_windows": 3200,
        "same_test_cases": 32,
        "test_npz_sha256": hashlib.sha256(args.test.read_bytes()).hexdigest(),
        "inference": "PyTorch FP32 on saved checkpoints",
        "comparison": results,
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, indent=2))
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
