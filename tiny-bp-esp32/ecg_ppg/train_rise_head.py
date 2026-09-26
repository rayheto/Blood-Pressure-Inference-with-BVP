"""Train a frozen-encoder residual head emphasizing rises from initial calibration."""

import argparse
import json
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.utils.tensorboard import SummaryWriter

from train_delta import DeltaBP, score
from train_fusion import load_matched
from train_pat_delta import checked_features, pair_features
from train_stream_bp import first_calibration_pairs


class RiseHead(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(73, 32), nn.ReLU(), nn.Linear(32, 16),
                                 nn.ReLU(), nn.Linear(16, 2))
        nn.init.zeros_(self.net[-1].weight)
        nn.init.zeros_(self.net[-1].bias)

    def forward(self, state):
        return torch.tanh(self.net(state)) * state.new_tensor([80., 40.])


def state_from_tensors(encoder, before, after, base, cuff, elapsed, extra):
    """Exactly the same feature construction is used during streaming inference."""
    with torch.no_grad():
        ea = encoder(before)
        eb = encoder(after)
    return torch.cat((eb - ea, eb, base / base.new_tensor([30., 15.]),
                      cuff / cuff.new_tensor([120., 80.]), elapsed / 1800, extra), dim=1)


def make_states(cache, model, x, features, y, times, pairs, device):
    if cache.exists():
        with np.load(cache) as z:
            return z["state"], z["base"], z["truth"], z["case"]
    out, bases = [], []
    for start in range(0, len(pairs), 768):
        rows = pairs[start:start + 768]
        a, b = rows.T
        before = torch.from_numpy(x[a]).to(device)
        after = torch.from_numpy(x[b]).to(device)
        elapsed = torch.from_numpy((times[b] - times[a]).astype(np.float32)[:, None]).to(device)
        cuff = torch.from_numpy(y[a].astype(np.float32)).to(device)
        extra = torch.from_numpy(pair_features(features, rows, True)[0]).to(device)
        with torch.no_grad():
            base = model(before, after, elapsed / 1800, extra)
            state = state_from_tensors(model.encoder, before, after, base, cuff, elapsed, extra)
        bases.append(base.cpu().numpy())
        out.append(state.cpu().numpy())
    state = np.concatenate(out)
    base = np.concatenate(bases)
    truth = y[pairs[:, 1]] - y[pairs[:, 0]]
    case = pairs[:, 0]
    np.savez(cache, state=state, base=base, truth=truth, case=case)
    return state, base, truth, case


def measure(pred, truth):
    result = score(pred, truth)
    for threshold in (20, 40, 60):
        mask = truth[:, 0] >= threshold
        result[f"rise_ge_{threshold}_n"] = int(mask.sum())
        result[f"rise_ge_{threshold}_sbp_mae"] = float(np.abs(pred[mask, 0] - truth[mask, 0]).mean()) if mask.any() else None
        result[f"rise_ge_{threshold}_sbp_bias"] = float((pred[mask, 0] - truth[mask, 0]).mean()) if mask.any() else None
    return result


def main():
    parser = argparse.ArgumentParser()
    for name in ("train", "test", "ecg-dir", "split", "preprocess", "base", "out", "tensorboard"):
        parser.add_argument("--" + name, type=Path, required=True)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--epochs", type=int, default=100)
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    torch.set_num_threads(4)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    split = json.loads(args.split.read_text())
    prep = json.loads(args.preprocess.read_text())
    ppg, _, y, subjects, cases, times, _ = load_matched(args.train, args.ecg_dir / "train")
    ppg_t, _, yt, _, cases_t, times_t, _ = load_matched(args.test, args.ecg_dir / "test")
    timing, timing_t = checked_features(args.ecg_dir / "timing_features_cache.npz", cases, times, cases_t, times_t)
    pairs = {
        "train": first_calibration_pairs(cases, times, np.isin(subjects, split["train"])),
        "val": first_calibration_pairs(cases, times, np.isin(subjects, split["validation"])),
        "test": first_calibration_pairs(cases_t, times_t),
    }
    x = ((ppg - prep["mean"]) / prep["std"]).astype(np.float32)[:, None, :]
    xt = ((ppg_t - prep["mean"]) / prep["std"]).astype(np.float32)[:, None, :]
    base_model = DeltaBP(1, 4).to(device).eval()
    base_model.load_state_dict(torch.load(args.base, map_location=device, weights_only=True))
    data = {}
    for name in ("train", "val", "test"):
        signal, f, labels, stamps = ((xt, timing_t, yt, times_t) if name == "test" else (x, timing, y, times))
        data[name] = make_states(args.out / f"{name}_state_cache.npz", base_model, signal, f,
                                 labels, stamps, pairs[name], device)
        print(name, len(pairs[name]), json.dumps(measure(data[name][1], data[name][2])), flush=True)
    state, base, truth, case = data["train"]
    state_val, base_val, truth_val, _ = data["val"]
    state_test, base_test, truth_test, _ = data["test"]
    bins = np.digitize(truth[:, 0], [-20, 20, 40])
    bin_counts = np.bincount(bins, minlength=4)
    _, case_inverse, case_counts = np.unique(case, return_inverse=True, return_counts=True)
    weights = 1 / np.sqrt(bin_counts[bins] * case_counts[case_inverse])
    weights = np.minimum(weights, np.median(weights) * 10)
    weights /= weights.sum()
    rng = np.random.default_rng(args.seed)
    head = RiseHead().to(device)
    optimizer = torch.optim.AdamW(head.parameters(), lr=.001, weight_decay=.005)
    states = torch.from_numpy(state).to(device)
    bases = torch.from_numpy(base).to(device)
    labels = torch.from_numpy(truth).to(device)
    val_states = torch.from_numpy(state_val).to(device)
    writer = SummaryWriter(str(args.tensorboard / "VitalDB_stream_rise_head"), flush_secs=5)
    best, best_epoch, stale = float("inf"), 0, 0
    for epoch in range(1, args.epochs + 1):
        head.train()
        order = rng.choice(len(state), len(state), replace=True, p=weights)
        for start in range(0, len(order), 1024):
            ids = torch.from_numpy(order[start:start + 1024]).to(device)
            correction = head(states[ids])
            error = (bases[ids] + correction - labels[ids]) / labels.new_tensor([20., 10.])
            huber = torch.where(error.abs() <= 1, .5 * error.square(), error.abs() - .5)
            rise_underestimate = (labels[ids, 0] >= 20) & (error[:, 0] < 0)
            huber[:, 0] *= torch.where(rise_underestimate, 1.5, 1.)
            loss = huber.mean() + .001 * ((correction / correction.new_tensor([40., 20.])).square()).mean()
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
        head.eval()
        with torch.no_grad():
            val_pred = base_val + head(val_states).cpu().numpy()
        vm = measure(val_pred, truth_val)
        selection = vm["sbp_change_mae"] + vm["dbp_change_mae"] + .35 * vm["rise_ge_40_sbp_mae"]
        for key in ("sbp_change_mae", "dbp_change_mae", "rise_ge_20_sbp_mae", "rise_ge_40_sbp_mae", "rise_ge_40_sbp_bias"):
            writer.add_scalar("validation/" + key, vm[key], epoch)
        writer.add_scalar("validation/selection", selection, epoch)
        writer.flush()
        if selection < best:
            best, best_epoch, stale = selection, epoch, 0
            torch.save(head.state_dict(), args.out / "rise_head.pt")
        else:
            stale += 1
        if epoch == 1 or epoch % 10 == 0 or stale >= 15:
            print("epoch", epoch, "selection", round(selection, 3), "SBP", round(vm["sbp_change_mae"], 3),
                  "rise40", round(vm["rise_ge_40_sbp_mae"], 3), flush=True)
        if stale >= 15:
            break
    head.load_state_dict(torch.load(args.out / "rise_head.pt", map_location=device, weights_only=True))
    head.eval()
    with torch.no_grad():
        vp = base_val + head(val_states).cpu().numpy()
        tp = base_test + head(torch.from_numpy(state_test).to(device)).cpu().numpy()
    np.save(args.out / "rise_head_test_predictions.npy", tp)
    report = {"seed": args.seed, "best_epoch": best_epoch, "train_pairs": len(state),
              "validation_pairs": len(state_val), "test_pairs": len(state_test),
              "train_rise_bins_lt_minus20_minus20to20_20to40_ge40": bin_counts.tolist(),
              "selection": "validation SBP MAE + DBP MAE + 0.35 * rise>=40 SBP MAE",
              "base_validation": measure(base_val, truth_val), "rise_validation": measure(vp, truth_val),
              "base_test": measure(base_test, truth_test), "rise_test": measure(tp, truth_test)}
    (args.out / "metrics.json").write_text(json.dumps(report, indent=2))
    writer.close()
    print(json.dumps(report, indent=2), flush=True)


if __name__ == "__main__":
    main()
