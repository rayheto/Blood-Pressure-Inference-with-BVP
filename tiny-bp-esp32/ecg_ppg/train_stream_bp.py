"""Train a calibration-based online BP updater from all available VitalDB windows.

The model consumes only a past calibration window and a completed current window.
It predicts the change from the calibration BP, with a 30-minute validity horizon.
"""

import argparse
import json
from pathlib import Path

import numpy as np
import torch
from torch.utils.tensorboard import SummaryWriter

from train_delta import DeltaBP, score
from train_fusion import load_matched
from train_pat_delta import checked_features, pair_features


HORIZON_S = 1800
TARGET_DELAYS_S = (10, 30, 60, 120, 300, 600, 1200, 1800)


def chronological_pairs(cases, times, allowed=None):
    """Past-to-future pairs across each case, preserving signal chronology."""
    pairs = set()
    for case in np.unique(cases):
        ids = np.flatnonzero(cases == case)
        if allowed is not None:
            ids = ids[allowed[ids]]
        if len(ids) < 2:
            continue
        ids = ids[np.argsort(times[ids])]
        stamps = times[ids]
        for pos, anchor in enumerate(ids[:-1]):
            for target in TARGET_DELAYS_S:
                wanted = stamps[pos] + target
                right = np.searchsorted(stamps, wanted, side="left")
                choices = (right - 1, right)
                valid = [k for k in choices if pos < k < len(ids)]
                if not valid:
                    continue
                k = min(valid, key=lambda k: abs(stamps[k] - wanted))
                gap = stamps[k] - stamps[pos]
                tolerance = max(10, target // 3)
                if 10 <= gap <= HORIZON_S and abs(gap - target) <= tolerance:
                    pairs.add((int(anchor), int(ids[k])))
    return np.asarray(sorted(pairs), np.int32).reshape(-1, 2)


def first_calibration_pairs(cases, times, allowed=None):
    """Only the earliest available window in each case is a calibration anchor."""
    pairs = []
    for case in np.unique(cases):
        ids = np.flatnonzero(cases == case)
        if allowed is not None:
            ids = ids[allowed[ids]]
        if len(ids) < 2:
            continue
        ids = ids[np.argsort(times[ids])]
        first = int(ids[0])
        pairs.extend((first, int(j)) for j in ids[1:]
                     if 10 <= times[j] - times[first] <= HORIZON_S)
    return np.asarray(pairs, np.int32).reshape(-1, 2)


def predict(model, x, features, y, times, pairs, use_ecg, batch_size, device):
    model.eval()
    pred = []
    with torch.no_grad():
        for start in range(0, len(pairs), batch_size):
            batch = pairs[start:start + batch_size]
            a, b = batch.T
            elapsed = ((times[b] - times[a]) / HORIZON_S).astype(np.float32)[:, None]
            extra = pair_features(features, batch, True)[0] if use_ecg else None
            pred.append(model(torch.from_numpy(x[a]).to(device),
                              torch.from_numpy(x[b]).to(device),
                              torch.from_numpy(elapsed).to(device),
                              torch.from_numpy(extra).to(device) if extra is not None else None).cpu().numpy())
    return np.concatenate(pred), y[pairs[:, 1]] - y[pairs[:, 0]]


def train_mode(name, x, y, times, features, train_pairs, val_pairs,
               xt, yt, tt, ft, test_pairs, args, meta):
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    rng = np.random.default_rng(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    use_ecg = name == "ppg_pat_rr"
    model = DeltaBP(1, 4 if use_ecg else 0).to(device)
    if args.init_dir is not None:
        model.load_state_dict(torch.load(args.init_dir / f"{name}.pt", map_location=device, weights_only=True))
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=3, factor=.5)
    writer = SummaryWriter(str(args.tensorboard / args.run_name / name), flush_secs=5)
    writer.add_text("run/description", json.dumps({**meta, "mode": name}, indent=2))
    writer.flush()
    # All eligible anchors are seen each epoch, but a different later window
    # is selected from their available chronological targets each time.
    anchors, starts, counts = np.unique(train_pairs[:, 0], return_index=True, return_counts=True)
    best, epoch_best, stale = float("inf"), 0, 0
    path = args.out / f"{name}.pt"
    for epoch in range(1, args.epochs + 1):
        if args.pairs == "first_calibration":
            selected = train_pairs
        else:
            choice = starts + (rng.random(len(anchors)) * counts).astype(np.int64)
            selected = train_pairs[choice]
        selected = selected[rng.permutation(len(selected))]
        model.train()
        total = 0.
        for start in range(0, len(selected), args.batch_size):
            batch = selected[start:start + args.batch_size]
            a, b = batch.T
            elapsed = ((times[b] - times[a]) / HORIZON_S).astype(np.float32)[:, None]
            extra = pair_features(features, batch, True)[0] if use_ecg else None
            target = torch.from_numpy((y[b] - y[a]).astype(np.float32)).to(device)
            optimizer.zero_grad(set_to_none=True)
            pred = model(torch.from_numpy(x[a]).to(device),
                         torch.from_numpy(x[b]).to(device),
                         torch.from_numpy(elapsed).to(device),
                         torch.from_numpy(extra).to(device) if extra is not None else None)
            loss = torch.mean(((pred - target) / 30) ** 2)
            loss.backward()
            optimizer.step()
            total += float(loss.detach()) * len(batch)
        val_pred, val_truth = predict(model, x, features, y, times, val_pairs, use_ecg,
                                     args.batch_size, device)
        val_mse = float(np.mean(((val_pred - val_truth) / 30) ** 2))
        scheduler.step(val_mse)
        val_score = score(val_pred, val_truth)
        writer.add_scalar("loss/train_mse_scaled", total / len(selected), epoch)
        writer.add_scalar("loss/val_mse_scaled", val_mse, epoch)
        writer.add_scalar("data/train_anchors_seen", len(anchors), epoch)
        writer.add_scalar("optimization/learning_rate", optimizer.param_groups[0]["lr"], epoch)
        for key, value in val_score.items():
            writer.add_scalar("validation/" + key, value, epoch)
        if val_mse < best:
            best, epoch_best, stale = val_mse, epoch, 0
            torch.save(model.state_dict(), path)
        else:
            stale += 1
        writer.flush()
        print(f"{name} epoch={epoch}/{args.epochs} train={total/len(selected):.5f} val={val_mse:.5f} "
              f"SBPΔMAE={val_score['sbp_change_mae']:.3f} DBPΔMAE={val_score['dbp_change_mae']:.3f}", flush=True)
        if stale >= args.patience:
            break
    model.load_state_dict(torch.load(path, map_location=device, weights_only=True))
    test_pred, test_truth = predict(model, xt, ft, yt, tt, test_pairs, use_ecg,
                                   args.batch_size, device)
    result = {"best_epoch": epoch_best, "test": score(test_pred, test_truth)}
    for key, value in result["test"].items():
        writer.add_scalar("test/" + key, value, epoch_best)
    writer.flush()
    writer.close()
    np.save(args.out / f"{name}_test_predictions.npy", test_pred)
    return result


def main():
    p = argparse.ArgumentParser()
    for key in ("train", "test", "ecg-dir", "split", "preprocess", "tensorboard", "out"):
        p.add_argument("--" + key, type=Path, required=True)
    p.add_argument("--run-name", default="VitalDB_stream_30min_all_windows")
    p.add_argument("--epochs", type=int, default=24)
    p.add_argument("--patience", type=int, default=6)
    p.add_argument("--batch-size", type=int, default=512)
    p.add_argument("--lr", type=float, default=.001)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--pairs", choices=("rotating", "first_calibration"), default="rotating")
    p.add_argument("--init-dir", type=Path)
    args = p.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    torch.set_num_threads(4)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    split = json.loads(args.split.read_text())
    prep = json.loads(args.preprocess.read_text())
    ppg, ecg, y, subjects, cases, times, _ = load_matched(args.train, args.ecg_dir / "train")
    ppg_t, ecg_t, yt, subjects_t, cases_t, tt, _ = load_matched(args.test, args.ecg_dir / "test")
    assert not (set(subjects) & set(subjects_t))
    assert set(subjects_t).issubset(set(split["test"]))
    features, ft = checked_features(args.ecg_dir / "timing_features_cache.npz", cases, times, cases_t, tt)
    pair_builder = first_calibration_pairs if args.pairs == "first_calibration" else chronological_pairs
    train_pairs = pair_builder(cases, times, np.isin(subjects, split["train"]))
    val_pairs = pair_builder(cases, times, np.isin(subjects, split["validation"]))
    test_pairs = pair_builder(cases_t, tt)
    assert len(train_pairs) and len(val_pairs) and len(test_pairs)
    x = ((ppg - prep["mean"]) / prep["std"]).astype(np.float32)[:, None, :]
    xt = ((ppg_t - prep["mean"]) / prep["std"]).astype(np.float32)[:, None, :]
    meta = {"source": "VitalDB matched ECG_II+PLETH+ART", "seed": args.seed,
            "target": "online cuff-calibrated BP change up to 30 minutes",
            "pair_sampling": args.pairs,
            "initialized_from": str(args.init_dir) if args.init_dir is not None else None,
            "window_samples": 1250, "sample_rate_hz": 125, "horizon_seconds": HORIZON_S,
            "train_pairs_available": len(train_pairs),
            "train_anchors_per_epoch": len(np.unique(train_pairs[:, 0])),
            "validation_pairs": len(val_pairs), "test_pairs": len(test_pairs),
            "train_cases": len(np.unique(cases[train_pairs[:, 0]])),
            "test_cases": len(np.unique(cases_t[test_pairs[:, 0]])),
            "model_inputs": ["past calibration PPG", "current PPG", "elapsed time",
                             "delta R-to-PPG-valley PAT proxy", "delta ECG RR", "PAT validity flags"]}
    print(json.dumps(meta, indent=2), flush=True)
    report = {"meta": meta, "models": {}}
    for name in ("ppg_only", "ppg_pat_rr"):
        report["models"][name] = train_mode(name, x, y, times, features, train_pairs, val_pairs,
                                            xt, yt, tt, ft, test_pairs, args, meta)
        (args.out / "metrics.json").write_text(json.dumps(report, indent=2))
    truth = yt[test_pairs[:, 1]] - yt[test_pairs[:, 0]]
    report["zero_change_mae_sbp_dbp"] = np.abs(truth).mean(axis=0).tolist()
    a = np.load(args.out / "ppg_only_test_predictions.npy")
    b = np.load(args.out / "ppg_pat_rr_test_predictions.npy")
    groups = [np.flatnonzero(cases_t[test_pairs[:, 0]] == c)
              for c in np.unique(cases_t[test_pairs[:, 0]])]
    rng = np.random.default_rng(2026)
    boot = []
    for _ in range(2000):
        ids = np.concatenate([groups[i] for i in rng.integers(len(groups), size=len(groups))])
        boot.append(np.abs(b[ids] - truth[ids]).mean(axis=0) -
                    np.abs(a[ids] - truth[ids]).mean(axis=0))
    report["pat_rr_minus_ppg_mae_95pct_sbp_dbp"] = np.quantile(boot, [.025, .975], axis=0).T.tolist()
    (args.out / "metrics.json").write_text(json.dumps(report, indent=2))
    print(json.dumps(report, indent=2), flush=True)


if __name__ == "__main__":
    main()
