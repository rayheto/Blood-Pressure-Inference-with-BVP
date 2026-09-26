"""Train a bounded supervised correction for the frozen streaming fusion model."""

import argparse
import json
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.utils.tensorboard import SummaryWriter

from train_delta import DeltaBP, score
from train_fusion import load_matched
from train_pat_delta import checked_features
from train_stream_bp import first_calibration_pairs, predict


class DynamicHead(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(5, 16), nn.Tanh(), nn.Linear(16, 2))
        nn.init.zeros_(self.net[-1].weight)
        nn.init.zeros_(self.net[-1].bias)

    def forward(self, features):
        return torch.tanh(self.net(features)) * torch.tensor([60., 30.], device=features.device)


def inputs(base, cuff, elapsed):
    return np.stack((cuff[:, 0] / 120, cuff[:, 1] / 80,
                     base[:, 0] / 40, base[:, 1] / 20, elapsed / 1800), axis=1).astype(np.float32)


def metrics(pred, truth, cuff):
    out = score(pred, truth)
    actual = truth + cuff
    estimated = pred + cuff
    out["sbp_bias"] = float(np.mean(estimated[:, 0] - actual[:, 0]))
    out["dbp_bias"] = float(np.mean(estimated[:, 1] - actual[:, 1]))
    for label, axis, low in (("sbp_ge_140", 0, 140), ("sbp_ge_160", 0, 160),
                             ("dbp_ge_90", 1, 90)):
        mask = actual[:, axis] >= low
        out[label + "_n"] = int(mask.sum())
        out[label + "_mae"] = float(np.mean(np.abs(estimated[mask, axis] - actual[mask, axis]))) if mask.any() else None
        out[label + "_bias"] = float(np.mean(estimated[mask, axis] - actual[mask, axis])) if mask.any() else None
    return out


def get_predictions(cache, model, x, features, y, times, pairs, device):
    if cache.exists():
        with np.load(cache) as z:
            pred, truth = z["pred"], z["truth"]
        if len(pred) != len(pairs):
            raise ValueError("Prediction cache pair count mismatch")
    else:
        pred, truth = predict(model, x, features, y, times, pairs, True, 1024, device)
        np.savez(cache, pred=pred, truth=truth)
    return pred, truth


def main():
    p = argparse.ArgumentParser()
    for arg in ("train", "test", "ecg-dir", "split", "preprocess", "base", "out", "tensorboard"):
        p.add_argument("--" + arg, type=Path, required=True)
    p.add_argument("--epochs", type=int, default=120)
    p.add_argument("--seed", type=int, default=42)
    args = p.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    torch.set_num_threads(4)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    split = json.loads(args.split.read_text())
    prep = json.loads(args.preprocess.read_text())
    ppg, _, y, subjects, cases, times, _ = load_matched(args.train, args.ecg_dir / "train")
    ppg_t, _, yt, _, cases_t, times_t, _ = load_matched(args.test, args.ecg_dir / "test")
    features, ft = checked_features(args.ecg_dir / "timing_features_cache.npz", cases, times, cases_t, times_t)
    train_pairs = first_calibration_pairs(cases, times, np.isin(subjects, split["train"]))
    val_pairs = first_calibration_pairs(cases, times, np.isin(subjects, split["validation"]))
    test_pairs = first_calibration_pairs(cases_t, times_t)
    x = ((ppg - prep["mean"]) / prep["std"]).astype(np.float32)[:, None, :]
    xt = ((ppg_t - prep["mean"]) / prep["std"]).astype(np.float32)[:, None, :]
    model = DeltaBP(1, 4).to(device).eval()
    model.load_state_dict(torch.load(args.base, map_location=device, weights_only=True))
    sets = {}
    for name, signal, timing, labels, stamps, pairs in (("train", x, features, y, times, train_pairs),
                                                        ("val", x, features, y, times, val_pairs),
                                                        ("test", xt, ft, yt, times_t, test_pairs)):
        base, truth = get_predictions(args.out / f"{name}_base_cache.npz", model, signal, timing, labels, stamps, pairs, device)
        cuff = labels[pairs[:, 0]]
        elapsed = stamps[pairs[:, 1]] - stamps[pairs[:, 0]]
        sets[name] = (inputs(base, cuff, elapsed), base, truth, cuff)
        print(name, len(pairs), json.dumps(metrics(base, truth, cuff)), flush=True)
    writer = SummaryWriter(str(args.tensorboard / "VitalDB_stream_dynamic_head"), flush_secs=5)
    report = {"seed": args.seed, "base": str(args.base), "train_pairs": len(train_pairs),
              "val_pairs": len(val_pairs), "test_pairs": len(test_pairs),
              "training": "frozen base; bounded 5-input, 16-hidden supervised residual head",
              "modes": {}}
    for mode in ("uniform", "high_pressure_weighted"):
        torch.manual_seed(args.seed)
        head = DynamicHead().to(device)
        optimizer = torch.optim.AdamW(head.parameters(), lr=.002, weight_decay=.001)
        xf, base, truth, cuff = sets["train"]
        xv, bv, tv, cv = sets["val"]
        xtest, bt, tt, ct = sets["test"]
        x_tensor = torch.from_numpy(xf).to(device)
        base_tensor = torch.from_numpy(base).to(device)
        truth_tensor = torch.from_numpy(truth).to(device)
        cuff_tensor = torch.from_numpy(cuff).to(device)
        xv_tensor = torch.from_numpy(xv).to(device)
        best, best_epoch, stale = float("inf"), 0, 0
        rng = np.random.default_rng(args.seed)
        for epoch in range(1, args.epochs + 1):
            head.train()
            order = rng.permutation(len(xf))
            for start in range(0, len(order), 1024):
                ids = torch.from_numpy(order[start:start + 1024]).to(device)
                correction = head(x_tensor[ids])
                predicted = base_tensor[ids] + correction
                error = predicted - truth_tensor[ids]
                scale = torch.tensor([20., 10.], device=device)
                scaled = error / scale
                huber = torch.where(scaled.abs() < 1, .5 * scaled.square(), scaled.abs() - .5)
                if mode == "high_pressure_weighted":
                    actual = truth_tensor[ids] + cuff_tensor[ids]
                    high_sbp = torch.clamp((actual[:, 0] - 130) / 30, 0, 1)
                    high_dbp = torch.clamp((actual[:, 1] - 80) / 20, 0, 1)
                    weight = torch.stack((1 + 2 * high_sbp, 1 + 2 * high_dbp), dim=1)
                    asym = torch.where((error < 0) & (weight > 1), 1.5, 1.)
                    huber = huber * weight * asym
                loss = huber.mean() + .0005 * ((correction / scale).square()).mean()
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                optimizer.step()
            head.eval()
            with torch.no_grad():
                pv = bv + head(xv_tensor).cpu().numpy()
            vm = metrics(pv, tv, cv)
            # Prespecified selection balances general error and high-pressure error.
            high = vm["sbp_ge_140_mae"] if vm["sbp_ge_140_mae"] is not None else vm["sbp_change_mae"]
            criterion = vm["sbp_change_mae"] + vm["dbp_change_mae"] + .5 * high
            for key in ("sbp_change_mae", "dbp_change_mae", "sbp_ge_140_mae", "sbp_bias", "dbp_bias"):
                if vm[key] is not None:
                    writer.add_scalar(f"{mode}/val/{key}", vm[key], epoch)
            writer.add_scalar(f"{mode}/val/selection_score", criterion, epoch)
            writer.flush()
            if criterion < best:
                best, best_epoch, stale = criterion, epoch, 0
                torch.save(head.state_dict(), args.out / f"{mode}.pt")
            else:
                stale += 1
            if epoch == 1 or epoch % 10 == 0 or stale >= 15:
                print(mode, epoch, "val", round(criterion, 3), "SBP", round(vm["sbp_change_mae"], 3),
                      "DBP", round(vm["dbp_change_mae"], 3), flush=True)
            if stale >= 15:
                break
        head.load_state_dict(torch.load(args.out / f"{mode}.pt", map_location=device, weights_only=True))
        head.eval()
        with torch.no_grad():
            pv = bv + head(xv_tensor).cpu().numpy()
            pt = bt + head(torch.from_numpy(xtest).to(device)).cpu().numpy()
        report["modes"][mode] = {"best_epoch": best_epoch, "validation": metrics(pv, tv, cv),
                                 "test": metrics(pt, tt, ct)}
        np.save(args.out / f"{mode}_test_predictions.npy", pt)
    report["base_test"] = metrics(sets["test"][1], sets["test"][2], sets["test"][3])
    (args.out / "metrics.json").write_text(json.dumps(report, indent=2))
    writer.close()
    print(json.dumps(report, indent=2), flush=True)


if __name__ == "__main__":
    main()
