"""Within-case BP change experiment on synchronized VitalDB ECG/PPG windows."""

import argparse
import json
import random
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.utils.tensorboard import SummaryWriter

from train_fusion import load_matched, prepare


class DeltaBP(nn.Module):
    def __init__(self, channels, extra_features=0):
        super().__init__()
        blocks = []
        for a, b in zip((channels, 8, 16, 24), (8, 16, 24, 32)):
            blocks += [nn.Conv1d(a, b, 9, 2, 4, bias=False), nn.BatchNorm1d(b), nn.ReLU()]
        self.encoder = nn.Sequential(*blocks, nn.AdaptiveAvgPool1d(1), nn.Flatten())
        self.head = nn.Sequential(nn.Linear(65 + extra_features, 32), nn.ReLU(), nn.Linear(32, 2))
        nn.init.zeros_(self.head[-1].weight)
        nn.init.zeros_(self.head[-1].bias)

    def forward(self, before, after, elapsed, extra=None):
        a = self.encoder(before)
        b = self.encoder(after)
        # Absolute starting BP is deliberately absent from the change predictor.
        pieces = (b - a, b, elapsed) if extra is None else (b - a, b, elapsed, extra)
        return self.head(torch.cat(pieces, dim=1)) * 30


def make_pairs(cases, times, allowed=None, targets=(60, 120, 180, 300)):
    """One nearest later window per target delay and anchor; deduplicate pairs."""
    pairs = set()
    for case in np.unique(cases):
        ids = np.flatnonzero(cases == case)
        if allowed is not None:
            ids = ids[allowed[ids]]
        if len(ids) < 2:
            continue
        ids = ids[np.argsort(times[ids])]
        stamps = times[ids]
        for position, anchor in enumerate(ids[:-1]):
            for delay in targets:
                wanted = stamps[position] + delay
                j = np.searchsorted(stamps, wanted, side="left")
                choices = [k for k in (j - 1, j) if k > position and k < len(ids)]
                if not choices:
                    continue
                k = min(choices, key=lambda k: abs(stamps[k] - wanted))
                gap = stamps[k] - stamps[position]
                if 60 <= gap <= 300 and abs(gap - delay) <= 25:
                    pairs.add((int(anchor), int(ids[k])))
    return np.asarray(sorted(pairs), dtype=np.int64).reshape(-1, 2)


def score(pred, truth):
    error = pred - truth
    result = {}
    for k, name in enumerate(("sbp", "dbp")):
        result[name + "_change_mae"] = float(np.mean(np.abs(error[:, k])))
        result[name + "_change_rmse"] = float(np.sqrt(np.mean(error[:, k] ** 2)))
        result[name + "_change_correlation"] = float(np.corrcoef(pred[:, k], truth[:, k])[0, 1]) if np.std(pred[:, k]) > 0 else 0.0
    return result


def predict(model, x, y, times, pairs, batch, device, channels):
    model.eval()
    output = []
    with torch.no_grad():
        for rows in np.array_split(pairs, max(1, (len(pairs) + batch - 1) // batch)):
            a, b = rows.T
            before = torch.from_numpy(x[a, :channels]).to(device)
            after = torch.from_numpy(x[b, :channels]).to(device)
            gap = torch.from_numpy(((times[b] - times[a]) / 300).astype(np.float32)[:, None]).to(device)
            output.append(model(before, after, gap).cpu().numpy())
    pred = np.concatenate(output)
    return pred, y[pairs[:, 1]] - y[pairs[:, 0]]


def train_one(name, x, y, times, train_pairs, val_pairs, test_x, test_y, test_times,
              test_pairs, args, metadata):
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    random.seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    channels = 1 if name in ("ppg_only", "ecg_only") else 2
    x = x[:, 1:2] if name == "ecg_only" else x[:, :channels]
    test_x = test_x[:, 1:2] if name == "ecg_only" else test_x[:, :channels]
    model = DeltaBP(channels).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    writer = SummaryWriter(str(args.tensorboard / args.run_name / name), flush_secs=5)
    writer.add_text("run/description", json.dumps({**metadata, "input": name, "target": "within-case 60-300 second BP change"}, indent=2))
    writer.flush()
    best, best_epoch = float("inf"), 0
    path = args.out / f"{name}.pt"
    for epoch in range(1, args.epochs + 1):
        model.train()
        order = np.random.permutation(len(train_pairs))
        total = 0.0
        for start in range(0, len(order), args.batch_size):
            rows = train_pairs[order[start:start + args.batch_size]]
            a, b = rows.T
            before = torch.from_numpy(x[a, :channels]).to(device)
            after = torch.from_numpy(x[b, :channels]).to(device)
            gap = torch.from_numpy(((times[b] - times[a]) / 300).astype(np.float32)[:, None]).to(device)
            target = torch.from_numpy((y[b] - y[a]).astype(np.float32)).to(device)
            optimizer.zero_grad(set_to_none=True)
            prediction = model(before, after, gap)
            loss = torch.mean(((prediction - target) / 30) ** 2)
            loss.backward()
            optimizer.step()
            total += float(loss.detach()) * len(rows)
        val_pred, val_truth = predict(model, x, y, times, val_pairs, args.batch_size, device, channels)
        val_mse = float(np.mean(((val_pred - val_truth) / 30) ** 2))
        val_score = score(val_pred, val_truth)
        writer.add_scalar("loss/train_mse_scaled", total / len(train_pairs), epoch)
        writer.add_scalar("loss/val_mse_scaled", val_mse, epoch)
        for key, value in val_score.items():
            writer.add_scalar("validation/" + key, value, epoch)
        if val_mse < best:
            best, best_epoch = val_mse, epoch
            torch.save(model.state_dict(), path)
        writer.flush()
        print(f"{name} epoch={epoch}/{args.epochs} val={val_mse:.4f} SBPΔMAE={val_score['sbp_change_mae']:.2f} DBPΔMAE={val_score['dbp_change_mae']:.2f}", flush=True)
    model.load_state_dict(torch.load(path, map_location=device, weights_only=True))
    test_pred, test_truth = predict(model, test_x, test_y, test_times, test_pairs, args.batch_size, device, channels)
    test_score = score(test_pred, test_truth)
    for key, value in test_score.items():
        writer.add_scalar("test/" + key, value, best_epoch)
    writer.flush()
    writer.close()
    np.save(args.out / f"{name}_test_predictions.npy", test_pred)
    return {"best_epoch": best_epoch, "validation_mse_scaled": best, "test": test_score}


def main():
    parser = argparse.ArgumentParser()
    for option in ("train", "test", "ecg-dir", "split", "preprocess", "tensorboard", "out"):
        parser.add_argument("--" + option, type=Path, required=True)
    parser.add_argument("--run-name", default="VitalDB_delta_60_300s")
    parser.add_argument("--epochs", type=int, default=12)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--max-train-pairs", type=int, default=60000)
    parser.add_argument("--lr", type=float, default=0.001)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    torch.set_num_threads(4)
    split = json.loads(args.split.read_text())
    prep = json.loads(args.preprocess.read_text())
    ppg, ecg, y, subjects, cases, times, _ = load_matched(args.train, args.ecg_dir / "train")
    ppg_t, ecg_t, y_t, subjects_t, cases_t, times_t, _ = load_matched(args.test, args.ecg_dir / "test")
    assert not (set(subjects) & set(subjects_t))
    assert set(subjects_t).issubset(set(split["test"]))
    train_mask = np.isin(subjects, split["train"])
    val_mask = np.isin(subjects, split["validation"])
    train_pairs = make_pairs(cases, times, train_mask)
    val_pairs = make_pairs(cases, times, val_mask)
    test_pairs = make_pairs(cases_t, times_t)
    assert len(train_pairs) and len(val_pairs) and len(test_pairs)
    rng = np.random.default_rng(args.seed)
    if len(train_pairs) > args.max_train_pairs:
        train_pairs = train_pairs[rng.choice(len(train_pairs), args.max_train_pairs, replace=False)]
    x, _ = prepare(ppg, ecg, prep["mean"], prep["std"], False)
    test_x, _ = prepare(ppg_t, ecg_t, prep["mean"], prep["std"], False)
    meta = {"source": "VitalDB matched ECG_II+PLETH+ART", "seed": args.seed,
            "delay_seconds": [60, 300], "train_pairs": len(train_pairs),
            "validation_pairs": len(val_pairs), "test_pairs": len(test_pairs),
            "train_cases": len(np.unique(cases[train_pairs[:, 0]])),
            "validation_cases": len(np.unique(cases[val_pairs[:, 0]])),
            "test_cases": len(np.unique(cases_t[test_pairs[:, 0]]))}
    print(json.dumps(meta, indent=2), flush=True)
    report = {"meta": meta, "models": {}}
    for name in ("ppg_only", "ecg_ppg"):
        report["models"][name] = train_one(name, x, y, times, train_pairs, val_pairs,
                                           test_x, y_t, times_t, test_pairs, args, meta)
        (args.out / "metrics.json").write_text(json.dumps(report, indent=2))
    a = np.load(args.out / "ppg_only_test_predictions.npy")
    b = np.load(args.out / "ecg_ppg_test_predictions.npy")
    truth = y_t[test_pairs[:, 1]] - y_t[test_pairs[:, 0]]
    pair_cases = cases_t[test_pairs[:, 0]]
    rng = np.random.default_rng(args.seed)
    unique = np.unique(pair_cases)
    groups = [np.flatnonzero(pair_cases == case) for case in unique]
    differences = []
    for _ in range(2000):
        selected = rng.integers(len(groups), size=len(groups))
        ids = np.concatenate([groups[i] for i in selected])
        differences.append(np.abs(b[ids] - truth[ids]).mean(axis=0) - np.abs(a[ids] - truth[ids]).mean(axis=0))
    report["ecg_minus_ppg_change_mae_95pct"] = np.quantile(differences, [.025, .975], axis=0).T.tolist()
    (args.out / "metrics.json").write_text(json.dumps(report, indent=2))
    print(json.dumps(report, indent=2), flush=True)


if __name__ == "__main__":
    main()
