"""Controlled PPG morphology ablation for the TinyBP architecture.

All transforms preserve the 1,250-sample length, detected peak times and
amplitudes. They alter waveform information and are not exact deletions of
published b-time/STT features; compare against the early-fall control.
"""

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch
from scipy.signal import find_peaks

from model import TinyBP


def transform_one(signal, mode):
    if mode == "raw":
        return signal.copy(), 0
    result = signal.copy()
    smooth = np.convolve(signal, np.ones(5) / 5, mode="same")
    prominence = max(1.0, float(np.std(signal) * 0.35))
    peaks, _ = find_peaks(smooth, distance=42, prominence=prominence)
    valleys, _ = find_peaks(-smooth, distance=42, prominence=prominence)
    changed = 0
    for peak_index, peak in enumerate(peaks):
        earlier = valleys[(valleys < peak) & (valleys >= peak - 100)]
        if len(earlier) == 0:
            continue
        foot = int(earlier[-1])
        if peak_index > 0 and foot <= peaks[peak_index - 1]:
            continue
        if peak - foot < 12 or signal[peak] - signal[foot] < prominence:
            continue
        if mode == "linear_rise":
            result[foot:peak + 1] = np.linspace(signal[foot], signal[peak], peak - foot + 1)
        elif mode == "fixed_rise":
            # Standardize rise to 20 samples and fill the original rise interval
            # with a plateau. The plateau still alters local timing information.
            start = max(foot, int(peak) - 20)
            result[foot:start] = signal[foot]
            result[start:peak + 1] = np.linspace(signal[foot], signal[peak], peak - start + 1)
        elif mode == "early_fall_control":
            # Same span length, after rather than before the peak.
            end = min(len(signal) - 1, int(peak) + (peak - foot))
            if peak_index + 1 < len(peaks):
                end = min(end, int(peaks[peak_index + 1]) - 1)
            if end <= peak + 1:
                continue
            result[peak:end + 1] = np.linspace(signal[peak], signal[end], end - peak + 1)
        changed += 1
    return result, changed


def transform_all(data, mode):
    if mode == "raw":
        return data, 0
    out = np.empty_like(data)
    changed = 0
    for i in range(len(data)):
        out[i], count = transform_one(data[i], mode)
        changed += count
        if (i + 1) % 50000 == 0:
            print(f"transform {mode}: {i + 1}/{len(data)}", flush=True)
    return out, changed


def predict(model, x, batch_size):
    model.eval()
    chunks = []
    with torch.no_grad():
        for batch in x.split(batch_size):
            chunks.append(model(batch).cpu().numpy())
    return np.concatenate(chunks)


def metrics(pred, label):
    e = pred - label
    return {
        "mae_sbp": float(np.abs(e[:, 0]).mean()),
        "mae_dbp": float(np.abs(e[:, 1]).mean()),
        "sd_sbp": float(e[:, 0].std(ddof=1)),
        "sd_dbp": float(e[:, 1].std(ddof=1)),
        "bias_sbp": float(e[:, 0].mean()),
        "bias_dbp": float(e[:, 1].mean()),
    }


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--train", type=Path, required=True)
    p.add_argument("--test", type=Path, required=True)
    p.add_argument("--split", type=Path, required=True)
    p.add_argument("--preprocess", type=Path, required=True)
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--mode", choices=("raw", "linear_rise", "fixed_rise", "early_fall_control"), required=True)
    p.add_argument("--epochs", type=int, default=12)
    p.add_argument("--batch-size", type=int, default=256)
    args = p.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    torch.manual_seed(42)
    np.random.seed(42)
    torch.set_num_threads(4)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    config = json.loads(args.preprocess.read_text())
    split = json.loads(args.split.read_text())
    with np.load(args.train, allow_pickle=False) as z:
        raw = z["ppg_signals"].astype(np.float32)
        labels = np.stack((z["sbp"], z["dbp"]), axis=1).astype(np.float32)
        subjects = z["subjects"].astype(str)
    with np.load(args.test, allow_pickle=False) as z:
        test_raw = z["ppg_signals"].astype(np.float32)
        test_labels = np.stack((z["sbp"], z["dbp"]), axis=1).astype(np.float32)
        test_subjects = z["subjects"].astype(str)
    train_idx = np.flatnonzero(np.isin(subjects, split["train"]))
    val_idx = np.flatnonzero(np.isin(subjects, split["validation"]))
    assert len(train_idx) == 307079 and len(val_idx) == 2600 and len(test_raw) == 3200
    assert set(test_subjects) == set(split["test"])
    assert set(subjects).isdisjoint(test_subjects)
    del subjects, test_subjects
    started = time.time()
    altered, changed_train = transform_all(raw, args.mode)
    del raw
    altered_test, changed_test = transform_all(test_raw, args.mode)
    del test_raw
    print(f"transformed {args.mode}: train beats={changed_train}, test beats={changed_test}, "
          f"seconds={time.time() - started:.1f}", flush=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    x = torch.from_numpy(((altered - config["mean"]) / config["std"]).astype(np.float32)[:, None, :]).to(device)
    x_test = torch.from_numpy(((altered_test - config["mean"]) / config["std"]).astype(np.float32)[:, None, :]).to(device)
    del altered, altered_test
    y = torch.from_numpy(labels).to(device)
    train_idx_device = torch.as_tensor(train_idx, device=device)
    val_idx_device = torch.as_tensor(val_idx, device=device)
    model = TinyBP().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=5, factor=0.5)
    best_loss = float("inf")
    best_epoch = 0
    history = []
    for epoch in range(1, args.epochs + 1):
        t = time.time()
        model.train()
        order = torch.randperm(len(train_idx), device=device)
        loss_sum = 0.0
        for batch in order.split(args.batch_size):
            ids = train_idx_device[batch]
            optimizer.zero_grad(set_to_none=True)
            loss = torch.nn.functional.mse_loss(model(x[ids]) / 100, y[ids] / 100)
            loss.backward()
            optimizer.step()
            loss_sum += float(loss.detach()) * len(batch)
        model.eval()
        with torch.no_grad():
            val_pred = predict(model, x[val_idx_device], args.batch_size)
        val_mse = float(np.mean(((val_pred - labels[val_idx]) / 100) ** 2))
        scheduler.step(val_mse)
        history.append({"epoch": epoch, "train_mse": loss_sum / len(train_idx), "val_mse": val_mse})
        if val_mse < best_loss:
            best_loss = val_mse
            best_epoch = epoch
            torch.save({k: v.detach().cpu().clone() for k, v in model.state_dict().items()},
                       args.out / f"{args.mode}.pt")
        print(f"{args.mode} epoch {epoch}/{args.epochs} train={history[-1]['train_mse']:.5f} "
              f"val={val_mse:.5f} seconds={time.time()-t:.1f}", flush=True)
    model.load_state_dict(torch.load(args.out / f"{args.mode}.pt", map_location=device,
                                     weights_only=True))
    test_pred = predict(model, x_test, args.batch_size)
    report = {
        "mode": args.mode,
        "train_windows": len(train_idx), "validation_windows": len(val_idx),
        "test_windows": len(test_labels), "epochs": args.epochs,
        "best_epoch": best_epoch,
        "changed_train_beats": changed_train, "changed_test_beats": changed_test,
        "test": metrics(test_pred, test_labels),
        "history": history,
    }
    (args.out / f"{args.mode}.json").write_text(json.dumps(report, indent=2))
    print(json.dumps({k: v for k, v in report.items() if k != "history"}, indent=2), flush=True)


if __name__ == "__main__":
    main()
