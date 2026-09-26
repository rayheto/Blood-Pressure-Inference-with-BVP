"""Add ECG-derived PAT and RR changes to the fixed VitalDB PPG delta pilot."""

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import torch
from torch.utils.tensorboard import SummaryWriter

from train_delta import DeltaBP, make_pairs, score
from train_fusion import load_matched


def checked_features(path, cases, times, test_cases, test_times):
    digest = hashlib.sha256()
    for ids, stamps in ((cases, times), (test_cases, test_times)):
        digest.update(ids.astype("U8").tobytes())
        digest.update(stamps.astype(np.int64).tobytes())
    with np.load(path, allow_pickle=False) as z:
        if str(z["key"]) != digest.hexdigest():
            raise ValueError("Timing-feature cache does not match the loaded windows")
        train = z["train"]
        test = z["test"]
    if train.shape != (len(cases), 3) or test.shape != (len(test_cases), 3):
        raise ValueError("Timing-feature array lengths do not match loaded windows")
    return train, test


def pair_features(features, pairs, include_rr):
    a, b = pairs.T
    valid_a = features[a, 2] >= .25  # at least 3 R-to-foot matches in 10 seconds
    valid_b = features[b, 2] >= .25
    valid = valid_a & valid_b
    pat_change = np.where(valid, features[b, 0] - features[a, 0], 0)
    columns = [pat_change, valid_a.astype(np.float32), valid_b.astype(np.float32)]
    if include_rr:
        columns.append(features[b, 1] - features[a, 1])
    return np.stack(columns, axis=1).astype(np.float32), valid


def infer(model, x, extra, times, pairs, batch_size, device):
    model.eval()
    pred = []
    with torch.no_grad():
        for start in range(0, len(pairs), batch_size):
            rows = pairs[start:start + batch_size]
            a, b = rows.T
            gap = ((times[b] - times[a]) / 300).astype(np.float32)[:, None]
            pred.append(model(torch.from_numpy(x[a]).to(device),
                              torch.from_numpy(x[b]).to(device),
                              torch.from_numpy(gap).to(device),
                              torch.from_numpy(extra[start:start + len(rows)]).to(device)).cpu().numpy())
    return np.concatenate(pred)


def fit(name, x, y, times, train_pairs, train_extra, val_pairs, val_extra,
        test_x, test_y, test_times, test_pairs, test_extra, args, meta):
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = DeltaBP(1, train_extra.shape[1]).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    writer = SummaryWriter(str(args.tensorboard / args.run_name / name), flush_secs=5)
    writer.add_text("run/description", json.dumps({**meta, "mode": name,
        "pat": "R-peak to PPG valley proxy; invalid pair imputed zero with validity flags"}, indent=2))
    writer.flush()
    best, best_epoch = float("inf"), 0
    path = args.out / f"{name}.pt"
    for epoch in range(1, args.epochs + 1):
        model.train()
        order = np.random.permutation(len(train_pairs))
        total = 0.
        for start in range(0, len(order), args.batch_size):
            sel = order[start:start + args.batch_size]
            a, b = train_pairs[sel].T
            gap = ((times[b] - times[a]) / 300).astype(np.float32)[:, None]
            target = torch.from_numpy((y[b] - y[a]).astype(np.float32)).to(device)
            optimizer.zero_grad(set_to_none=True)
            pred = model(torch.from_numpy(x[a]).to(device),
                         torch.from_numpy(x[b]).to(device),
                         torch.from_numpy(gap).to(device),
                         torch.from_numpy(train_extra[sel]).to(device))
            loss = torch.mean(((pred - target) / 30) ** 2)
            loss.backward()
            optimizer.step()
            total += float(loss.detach()) * len(sel)
        vp = infer(model, x, val_extra, times, val_pairs, args.batch_size, device)
        vy = y[val_pairs[:, 1]] - y[val_pairs[:, 0]]
        mse = float(np.mean(((vp - vy) / 30) ** 2))
        metrics = score(vp, vy)
        writer.add_scalar("loss/train_mse_scaled", total / len(train_pairs), epoch)
        writer.add_scalar("loss/val_mse_scaled", mse, epoch)
        for key, value in metrics.items():
            writer.add_scalar("validation/" + key, value, epoch)
        if mse < best:
            best, best_epoch = mse, epoch
            torch.save(model.state_dict(), path)
        writer.flush()
        print(f"{name} epoch={epoch}/{args.epochs} val={mse:.4f} SBPΔMAE={metrics['sbp_change_mae']:.2f} DBPΔMAE={metrics['dbp_change_mae']:.2f}", flush=True)
    model.load_state_dict(torch.load(path, map_location=device, weights_only=True))
    test_pred = infer(model, test_x, test_extra, test_times, test_pairs, args.batch_size, device)
    truth = test_y[test_pairs[:, 1]] - test_y[test_pairs[:, 0]]
    result = {"best_epoch": best_epoch, "validation_mse_scaled": best,
              "test": score(test_pred, truth)}
    for key, value in result["test"].items():
        writer.add_scalar("test/" + key, value, best_epoch)
    writer.flush()
    writer.close()
    np.save(args.out / f"{name}_test_predictions.npy", test_pred)
    return result


def main():
    p = argparse.ArgumentParser()
    for key in ("train", "test", "ecg-dir", "split", "preprocess", "tensorboard", "out", "baseline-run"):
        p.add_argument("--" + key, type=Path, required=True)
    p.add_argument("--run-name", default="VitalDB_delta_PAT_60_300s")
    p.add_argument("--epochs", type=int, default=12)
    p.add_argument("--batch-size", type=int, default=256)
    p.add_argument("--max-train-pairs", type=int, default=60000)
    p.add_argument("--lr", type=float, default=.001)
    p.add_argument("--seed", type=int, default=42)
    args = p.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    torch.set_num_threads(4)
    split = json.loads(args.split.read_text())
    prep = json.loads(args.preprocess.read_text())
    ppg, ecg, y, subjects, cases, times, _ = load_matched(args.train, args.ecg_dir / "train")
    ppg_t, ecg_t, y_t, subjects_t, cases_t, times_t, _ = load_matched(args.test, args.ecg_dir / "test")
    assert not (set(subjects) & set(subjects_t))
    assert set(subjects_t).issubset(set(split["test"]))
    features, test_features = checked_features(args.ecg_dir / "timing_features_cache.npz", cases, times, cases_t, times_t)
    train_pairs = make_pairs(cases, times, np.isin(subjects, split["train"]))
    val_pairs = make_pairs(cases, times, np.isin(subjects, split["validation"]))
    test_pairs = make_pairs(cases_t, times_t)
    rng = np.random.default_rng(args.seed)
    if len(train_pairs) > args.max_train_pairs:
        train_pairs = train_pairs[rng.choice(len(train_pairs), args.max_train_pairs, replace=False)]
    baseline = json.loads((args.baseline_run / "metrics.json").read_text())["meta"]
    for key, value in (("train_pairs", len(train_pairs)), ("validation_pairs", len(val_pairs)),
                       ("test_pairs", len(test_pairs)), ("seed", args.seed)):
        if baseline[key] != value:
            raise ValueError(f"Baseline comparison mismatch: {key}")
    x = ((ppg - prep["mean"]) / prep["std"]).astype(np.float32)[:, None, :]
    xt = ((ppg_t - prep["mean"]) / prep["std"]).astype(np.float32)[:, None, :]
    meta = {"source": "VitalDB ECG_II+PLETH+ART", "seed": args.seed,
            "train_pairs": len(train_pairs), "validation_pairs": len(val_pairs),
            "test_pairs": len(test_pairs), "test_cases": len(np.unique(cases_t[test_pairs[:, 0]])),
            "pat_proxy_validity_threshold": "at least 3 R-foot matches in both windows"}
    print(json.dumps(meta, indent=2), flush=True)
    report = {"meta": meta, "models": {}}
    truth = y_t[test_pairs[:, 1]] - y_t[test_pairs[:, 0]]
    baseline_pred = np.load(args.baseline_run / "ppg_only_test_predictions.npy")
    if baseline_pred.shape != truth.shape:
        raise ValueError("Baseline predictions do not match test pairs")
    pair_cases = cases_t[test_pairs[:, 0]]
    for name, include_rr in (("ppg_pat", False), ("ppg_pat_rr", True)):
        te, tv = pair_features(features, train_pairs, include_rr)
        ve, vv = pair_features(features, val_pairs, include_rr)
        ee, ev = pair_features(test_features, test_pairs, include_rr)
        report["meta"]["pat_valid_fraction_train"] = float(tv.mean())
        report["meta"]["pat_valid_fraction_validation"] = float(vv.mean())
        report["meta"]["pat_valid_fraction_test"] = float(ev.mean())
        result = fit(name, x, y, times, train_pairs, te, val_pairs, ve,
                     xt, y_t, times_t, test_pairs, ee, args, meta)
        candidate = np.load(args.out / f"{name}_test_predictions.npy")
        result["test_pat_valid_pairs"] = int(ev.sum())
        result["test_pat_valid_subset"] = score(candidate[ev], truth[ev])
        result["ppg_baseline_pat_valid_subset"] = score(baseline_pred[ev], truth[ev])
        groups = [np.flatnonzero(pair_cases == case) for case in np.unique(pair_cases)]
        bootstrap = []
        boot_rng = np.random.default_rng(args.seed)
        for _ in range(2000):
            selected = boot_rng.integers(len(groups), size=len(groups))
            ids = np.concatenate([groups[i] for i in selected])
            bootstrap.append(np.abs(candidate[ids] - truth[ids]).mean(axis=0) -
                             np.abs(baseline_pred[ids] - truth[ids]).mean(axis=0))
        result["minus_ppg_change_mae_95pct_sbp_dbp"] = np.quantile(bootstrap, [.025, .975], axis=0).T.tolist()
        report["models"][name] = result
        (args.out / "metrics.json").write_text(json.dumps(report, indent=2))
    print(json.dumps(report, indent=2), flush=True)


if __name__ == "__main__":
    main()
