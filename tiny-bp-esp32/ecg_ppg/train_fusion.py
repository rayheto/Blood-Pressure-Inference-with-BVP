"""Matched-window PPG, ECG+PPG, and ECG+PPG+PAT comparison.

All three modes use only windows with a validated synchronized ECG match.
TensorBoard is updated and flushed after every epoch.
"""

import argparse
import hashlib
import json
import random
from pathlib import Path

import numpy as np
import torch
from scipy.signal import butter, find_peaks, sosfiltfilt
from torch import nn
from torch.utils.tensorboard import SummaryWriter


class FusionBP(nn.Module):
    def __init__(self, channels, extra=0):
        super().__init__()
        layers = []
        for a, b in zip((channels, 8, 16, 24), (8, 16, 24, 32)):
            layers += [nn.Conv1d(a, b, 9, stride=2, padding=4, bias=False),
                       nn.BatchNorm1d(b), nn.ReLU()]
        self.features = nn.Sequential(*layers)
        self.pool = nn.AdaptiveAvgPool1d(1)
        self.head = nn.Linear(32 + extra, 2)
        with torch.no_grad():
            self.head.bias.copy_(torch.tensor([1.2, 0.8]))

    def forward(self, x, extra=None):
        x = self.pool(self.features(x)).flatten(1)
        if extra is not None:
            x = torch.cat((x, extra), dim=1)
        return self.head(x) * 100


class ResidualFusionBP(nn.Module):
    """Frozen PPG baseline plus a learnable ECG correction, initially zero."""

    def __init__(self, baseline_weights, extra=0):
        super().__init__()
        self.baseline = FusionBP(1)
        self.baseline.load_state_dict(torch.load(baseline_weights, map_location="cpu", weights_only=True))
        for parameter in self.baseline.parameters():
            parameter.requires_grad_(False)
        self.ecg = nn.Sequential(
            nn.Conv1d(1, 8, 9, stride=2, padding=4, bias=False), nn.BatchNorm1d(8), nn.ReLU(),
            nn.Conv1d(8, 16, 9, stride=2, padding=4, bias=False), nn.BatchNorm1d(16), nn.ReLU(),
            nn.Conv1d(16, 16, 9, stride=2, padding=4, bias=False), nn.BatchNorm1d(16), nn.ReLU(),
            nn.AdaptiveAvgPool1d(1), nn.Flatten())
        self.head = nn.Linear(16 + extra, 2)
        nn.init.zeros_(self.head.weight)
        nn.init.zeros_(self.head.bias)

    def train(self, mode=True):
        super().train(mode)
        self.baseline.eval()
        return self

    def forward(self, x, extra=None):
        with torch.no_grad():
            base = self.baseline(x[:, :1])
        e = self.ecg(x[:, 1:2])
        if extra is not None:
            e = torch.cat((e, extra), dim=1)
        return base + self.head(e) * 100


def load_matched(path, folder):
    with np.load(path, allow_pickle=False) as z:
        source = {k: z[k] for k in ("ppg_signals", "sbp", "dbp", "subjects", "caseids", "time_s")}
    rows, ecg, correlations = [], [], []
    for file in folder.glob("*.npz"):
        with np.load(file, allow_pickle=False) as z:
            file_rows = z["row_index"]
            if not np.all(source["caseids"][file_rows].astype(str) == file.stem):
                raise ValueError(f"Case mismatch in {file}")
            rows.append(file_rows)
            ecg.append(z["ecg_signals"])
            correlations.append(z["ppg_match_correlation"])
    if not rows:
        raise ValueError(f"No matched ECG files in {folder}")
    order = np.argsort(np.concatenate(rows))
    indices = np.concatenate(rows)[order]
    if len(np.unique(indices)) != len(indices):
        raise ValueError("Duplicate source rows")
    ecg = np.concatenate(ecg)[order]
    correlations = np.concatenate(correlations)[order]
    ppg = source["ppg_signals"][indices]
    y = np.stack((source["sbp"][indices], source["dbp"][indices]), axis=1)
    return ppg, ecg, y, source["subjects"][indices].astype(str), source["caseids"][indices].astype(str), source["time_s"][indices], correlations


def signal_features(ppg, ecg, sos):
    """Three input features: median ECG-to-PPG-foot delay, ECG RR, match count.

    Delays outside 80-800 ms are excluded. Values are standardized to a
    stable engineering scale; missing delay is imputed to 350 ms with zero
    match-count feature. This is a simple PAT proxy, not a validated PTT.
    """
    rate = 125
    filtered = sosfiltfilt(sos, ecg)
    # Lead II polarity varies. Squared QRS energy is polarity-independent.
    energy = filtered * filtered
    r, _ = find_peaks(energy, distance=round(0.35 * rate),
                      prominence=max(float(np.percentile(energy, 95)) * 0.35, 0.002))
    if len(r) >= 2:
        rr = np.diff(r) / rate
        rr = rr[(rr >= 0.35) & (rr <= 1.5)]
        rr_median = float(np.median(rr)) if len(rr) else 0.8
    else:
        rr_median = 0.8
    smooth = np.convolve(ppg, np.ones(5) / 5, mode="same")
    valleys, _ = find_peaks(-smooth, distance=40,
                            prominence=max(1.0, float(np.std(ppg) * 0.35)))
    delays = []
    for foot in valleys:
        earlier = r[(r <= foot - 10) & (r >= foot - 100)]
        if len(earlier):
            delays.append((foot - earlier[-1]) / rate)
    pat = float(np.median(delays)) if len(delays) >= 3 else 0.35
    return np.array([(pat - 0.35) / 0.15,
                     (rr_median - 0.8) / 0.25,
                     min(len(delays), 12) / 12], dtype=np.float32)


def prepare(ppg, ecg, ppg_mean, ppg_std, with_features):
    p = ((ppg - ppg_mean) / ppg_std).astype(np.float32)
    median = np.median(ecg, axis=1, keepdims=True)
    spread = np.maximum(np.std(ecg, axis=1, keepdims=True), 0.03)
    e = np.clip((ecg - median) / spread, -8, 8).astype(np.float32)
    x = np.stack((p, e), axis=1)
    features = None
    if with_features:
        sos = butter(2, (5, 20), btype="bandpass", fs=125, output="sos")
        features = np.stack([signal_features(a, b, sos) for a, b in zip(ppg, ecg)])
    return x, features


def extract_features(ppg, ecg):
    sos = butter(2, (5, 20), btype="bandpass", fs=125, output="sos")
    output = []
    for i, (a, b) in enumerate(zip(ppg, ecg), 1):
        output.append(signal_features(a, b, sos))
        if i % 10000 == 0:
            print(f"timing features {i}/{len(ppg)}", flush=True)
    return np.stack(output)


def load_or_extract_features(cache_path, ppg, ecg, ppg_test, ecg_test, cases,
                             times, test_cases, test_times):
    digest = hashlib.sha256()
    for ids, stamps in ((cases, times), (test_cases, test_times)):
        digest.update(ids.astype("U8").tobytes())
        digest.update(stamps.astype(np.int64).tobytes())
    key = digest.hexdigest()
    if cache_path.exists():
        with np.load(cache_path, allow_pickle=False) as z:
            if str(z["key"]) == key:
                print(f"loaded timing features from {cache_path}", flush=True)
                return z["train"], z["test"]
    train = extract_features(ppg, ecg)
    test = extract_features(ppg_test, ecg_test)
    np.savez_compressed(cache_path, key=key, train=train, test=test)
    return train, test


def predict(model, x, features, batch_size, device):
    model.eval()
    out = []
    with torch.no_grad():
        for start in range(0, len(x), batch_size):
            xb = x[start:start + batch_size].to(device)
            fb = features[start:start + batch_size].to(device) if features is not None else None
            out.append(model(xb, fb).cpu().numpy())
    return np.concatenate(out)


def scores(pred, y):
    err = pred - y
    return {"sbp_mae": float(np.abs(err[:, 0]).mean()),
            "dbp_mae": float(np.abs(err[:, 1]).mean()),
            "sbp_bias": float(err[:, 0].mean()),
            "dbp_bias": float(err[:, 1].mean()),
            "sbp_sd": float(err[:, 0].std(ddof=1)),
            "dbp_sd": float(err[:, 1].std(ddof=1))}


def train_mode(mode, x_pool, x_test, feat_pool, feat_test, y_pool, y_test,
               train_idx, val_idx, args, meta):
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    random.seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    channels = 1 if mode == "ppg_only" else 2
    use_feat = mode in ("ecg_ppg_pat", "ecg_delay_residual")
    residual = mode in ("ecg_residual", "ecg_delay_residual")
    x = torch.from_numpy(x_pool[:, :channels]).to(device)
    xt = torch.from_numpy(x_test[:, :channels])
    f = torch.from_numpy(feat_pool).to(device) if use_feat else None
    ft = torch.from_numpy(feat_test) if use_feat else None
    y = torch.from_numpy(y_pool).to(device)
    model = (ResidualFusionBP(args.out / "ppg_only.pt", 3 if use_feat else 0)
             if residual else FusionBP(channels, 3 if use_feat else 0)).to(device)
    opt = torch.optim.Adam((p for p in model.parameters() if p.requires_grad), lr=0.001)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(opt, patience=5, factor=0.5)
    train_idx = torch.from_numpy(train_idx).to(device)
    val_idx = torch.from_numpy(val_idx).to(device)
    writer = SummaryWriter(str(args.tensorboard / args.run_name / mode), flush_secs=5)
    writer.add_text("run/description", json.dumps({**meta, "mode": mode,
        "input": "PPG" if channels == 1 else "synchronized PPG+ECG",
        "explicit_features": "PAT proxy, RR, valid pair count" if use_feat else "none"}, indent=2))
    writer.add_scalar("data/train_windows", meta["train_windows"], 0)
    writer.add_scalar("data/validation_windows", meta["validation_windows"], 0)
    writer.add_scalar("data/test_windows", meta["test_windows"], 0)
    writer.flush()
    best = float("inf")
    best_epoch = 0
    path = args.out / f"{mode}.pt"
    if residual:
        initial_pred = predict(model, x[val_idx].cpu(), f[val_idx].cpu() if use_feat else None,
                               args.batch_size, device)
        initial_y = y[val_idx].cpu().numpy()
        best = float(np.mean(((initial_pred - initial_y) / 100) ** 2))
        initial_score = scores(initial_pred, initial_y)
        writer.add_scalar("loss/val_mse_scaled", best, 0)
        writer.add_scalar("mae/val_sbp_mmhg", initial_score["sbp_mae"], 0)
        writer.add_scalar("mae/val_dbp_mmhg", initial_score["dbp_mae"], 0)
        torch.save({k: v.detach().cpu() for k, v in model.state_dict().items()}, path)
        writer.flush()
    for epoch in range(1, args.epochs + 1):
        model.train()
        order = train_idx[torch.randperm(len(train_idx), device=device)]
        running = 0.0
        for ids in order.split(args.batch_size):
            opt.zero_grad(set_to_none=True)
            pred = model(x[ids], f[ids] if use_feat else None)
            loss = torch.nn.functional.mse_loss(pred / 100, y[ids] / 100)
            loss.backward()
            opt.step()
            running += float(loss.detach()) * len(ids)
        val_pred = predict(model, x[val_idx].cpu(), f[val_idx].cpu() if use_feat else None,
                           args.batch_size, device)
        val_y = y[val_idx].cpu().numpy()
        val_mse = float(np.mean(((val_pred - val_y) / 100) ** 2))
        scheduler.step(val_mse)
        s = scores(val_pred, val_y)
        writer.add_scalar("loss/train_mse_scaled", running / len(train_idx), epoch)
        writer.add_scalar("loss/val_mse_scaled", val_mse, epoch)
        writer.add_scalar("mae/val_sbp_mmhg", s["sbp_mae"], epoch)
        writer.add_scalar("mae/val_dbp_mmhg", s["dbp_mae"], epoch)
        writer.add_scalar("optimization/learning_rate", opt.param_groups[0]["lr"], epoch)
        writer.flush()
        if val_mse < best:
            best, best_epoch = val_mse, epoch
            torch.save({k: v.detach().cpu() for k, v in model.state_dict().items()}, path)
        writer.add_scalar("loss/best_val_mse_scaled", best, epoch)
        writer.flush()
        print(f"{mode} epoch={epoch}/{args.epochs} train={running/len(train_idx):.5f} "
              f"val={val_mse:.5f} SBP_MAE={s['sbp_mae']:.3f} DBP_MAE={s['dbp_mae']:.3f}", flush=True)
    model.load_state_dict(torch.load(path, map_location=device, weights_only=True))
    test_pred = predict(model, xt, ft, args.batch_size, device)
    test_score = scores(test_pred, y_test)
    for key, value in test_score.items():
        writer.add_scalar("test/" + key, value, best_epoch)
    writer.add_text("run/best_checkpoint", f"epoch={best_epoch}; val MSE={best:.6f}")
    writer.flush()
    writer.close()
    np.save(args.out / f"{mode}_test_predictions.npy", test_pred)
    return {"best_epoch": best_epoch, "test": test_score}


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--train", type=Path, required=True)
    p.add_argument("--test", type=Path, required=True)
    p.add_argument("--ecg-dir", type=Path, required=True)
    p.add_argument("--split", type=Path, required=True)
    p.add_argument("--preprocess", type=Path, required=True)
    p.add_argument("--tensorboard", type=Path, required=True)
    p.add_argument("--run-name", required=True)
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--epochs", type=int, default=20)
    p.add_argument("--batch-size", type=int, default=256)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--modes", nargs="+", choices=("ppg_only", "ecg_ppg", "ecg_ppg_pat",
                                                "ecg_residual", "ecg_delay_residual"),
                   default=("ppg_only", "ecg_ppg", "ecg_ppg_pat"))
    args = p.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    torch.set_num_threads(4)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    split = json.loads(args.split.read_text())
    config = json.loads(args.preprocess.read_text())
    ppg, ecg, y, subjects, cases, times, corr = load_matched(args.train, args.ecg_dir / "train")
    ppg_t, ecg_t, y_t, subjects_t, cases_t, times_t, corr_t = load_matched(args.test, args.ecg_dir / "test")
    assert len(set(subjects) & set(subjects_t)) == 0
    assert set(subjects_t).issubset(set(split["test"]))
    train_idx = np.flatnonzero(np.isin(subjects, split["train"]))
    val_idx = np.flatnonzero(np.isin(subjects, split["validation"]))
    assert len(train_idx) and len(val_idx)
    meta = {"run_name": args.run_name, "train_windows": len(train_idx),
            "validation_windows": len(val_idx), "test_windows": len(y_t),
            "train_subjects": len(np.unique(subjects[train_idx])),
            "validation_subjects": len(np.unique(subjects[val_idx])),
            "test_subjects": len(np.unique(subjects_t)),
            "min_ppg_match_correlation": float(min(corr.min(), corr_t.min())),
            "epochs": args.epochs, "seed": args.seed,
            "source": "VitalDB SNUADC/PLETH and synchronized SNUADC/ECG_II"}
    print(json.dumps(meta, indent=2), flush=True)
    x, feature = prepare(ppg, ecg, config["mean"], config["std"], False)
    xt, feature_t = prepare(ppg_t, ecg_t, config["mean"], config["std"], False)
    report_path = args.out / "metrics.json"
    report = json.loads(report_path.read_text()) if report_path.exists() else {"meta": meta, "models": {}}
    if report["meta"] != meta:
        raise ValueError("Existing metrics use a different matched-window snapshot or run settings")
    for mode in args.modes:
        if mode in ("ecg_ppg_pat", "ecg_delay_residual") and feature is None:
            feature, feature_t = load_or_extract_features(
                args.ecg_dir / "timing_features_cache.npz", ppg, ecg, ppg_t, ecg_t,
                cases, times, cases_t, times_t)
        report["models"][mode] = train_mode(mode, x, xt, feature, feature_t,
                                               y, y_t, train_idx, val_idx, args, meta)
        report_path.write_text(json.dumps(report, indent=2))
    print(json.dumps(report, indent=2), flush=True)


if __name__ == "__main__":
    main()
