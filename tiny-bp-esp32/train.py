"""Train and export TinyBP. Expects PulseDB-style subset NPZ files."""

import argparse
import json
import random
import sys
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset
from torch.utils.tensorboard import SummaryWriter

from model import TinyBP, TinyBPExport


class Windows(Dataset):
    def __init__(self, signals, targets, indices, mean, std):
        self.signals = signals
        self.targets = targets
        self.indices = np.asarray(indices, dtype=np.int64)
        self.mean = mean
        self.std = std

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, index):
        i = self.indices[index]
        signal = np.asarray(self.signals[i], dtype=np.float32)
        if signal.shape != (1250,):
            raise ValueError(f"PPG window {i} has shape {signal.shape}; expected (1250,)")
        x = torch.from_numpy(((signal - self.mean) / self.std)[None, :].copy())
        y = torch.from_numpy(self.targets[i].copy())
        return x, y


def load_subset(path):
    with np.load(path, allow_pickle=False) as data:
        required = {"ppg_signals", "sbp", "dbp", "subjects"}
        missing = required - set(data.files)
        if missing:
            raise ValueError(f"{path}: missing {sorted(missing)}")
        signals = np.asarray(data["ppg_signals"])
        labels = np.stack((data["sbp"], data["dbp"]), axis=1).astype(np.float32)
        subjects = np.asarray(data["subjects"]).astype(str)
    if signals.ndim != 2 or signals.shape[1] != 1250:
        raise ValueError(f"{path}: expected PPG array (N, 1250), got {signals.shape}")
    if not (len(signals) == len(labels) == len(subjects)):
        raise ValueError("Signal, label, and subject counts differ")
    if not np.isfinite(labels).all() or not np.isfinite(signals).all():
        raise ValueError("Nonfinite signal or label")
    return signals, labels, subjects


def split_by_subject(subjects, validation_fraction, seed):
    unique = np.unique(subjects)
    if len(unique) < 2:
        raise ValueError("Need at least two training subjects for independent validation")
    rng = np.random.default_rng(seed)
    rng.shuffle(unique)
    n_val = min(len(unique) - 1, max(1, round(len(unique) * validation_fraction)))
    val_subjects = set(unique[:n_val])
    val_mask = np.fromiter((s in val_subjects for s in subjects), dtype=bool)
    return np.flatnonzero(~val_mask), np.flatnonzero(val_mask)


def score(model, loader, device):
    model.eval()
    errors = []
    with torch.no_grad():
        for x, y in loader:
            pred = model(x.to(device)).cpu().numpy()
            errors.append(pred - y.numpy())
    error = np.concatenate(errors)
    return {
        "mae_sbp": float(np.abs(error[:, 0]).mean()),
        "mae_dbp": float(np.abs(error[:, 1]).mean()),
        "bias_sbp": float(error[:, 0].mean()),
        "bias_dbp": float(error[:, 1].mean()),
        "sd_sbp": float(error[:, 0].std(ddof=1)) if len(error) > 1 else None,
        "sd_dbp": float(error[:, 1].std(ddof=1)) if len(error) > 1 else None,
        "windows": len(error),
    }


def trend_score(model, test_path, signals, labels, mean, std, batch_size):
    """Descriptive within-case change tracking; no intervention attribution."""
    with np.load(test_path, allow_pickle=False) as data:
        if not {"caseids", "time_s"}.issubset(data.files):
            return None
        caseids = np.asarray(data["caseids"]).astype(str)
        times = np.asarray(data["time_s"], dtype=np.int64)
    loader = DataLoader(Windows(signals, labels, np.arange(len(signals)), mean, std),
                        batch_size=batch_size)
    with torch.no_grad():
        predictions = np.concatenate([model(x).numpy() for x, _ in loader])
    pairs = []
    for caseid in np.unique(caseids):
        indices = np.flatnonzero(caseids == caseid)
        indices = indices[np.argsort(times[indices])]
        for pos, i in enumerate(indices[:-1]):
            later = indices[pos + 1:]
            delays = times[later] - times[i]
            eligible = later[(delays >= 60) & (delays <= 300)]
            if len(eligible):
                j = eligible[0]
                pairs.append((labels[j] - labels[i], predictions[j] - predictions[i]))
    if not pairs:
        return None
    truth = np.stack([pair[0] for pair in pairs])
    prediction = np.stack([pair[1] for pair in pairs])
    result = {"pairs": len(pairs), "separation_s": "60-300"}
    for col, name in enumerate(("sbp", "dbp")):
        large = np.abs(truth[:, col]) >= 10
        result[name] = {
            "delta_mae": float(np.abs(prediction[:, col] - truth[:, col]).mean()),
            "delta_correlation": float(np.corrcoef(truth[:, col], prediction[:, col])[0, 1])
            if np.std(truth[:, col]) > 0 and np.std(prediction[:, col]) > 0 else None,
            "large_change_pairs": int(large.sum()),
            "large_change_direction_accuracy":
                float(np.mean(np.sign(prediction[large, col]) == np.sign(truth[large, col])))
                if large.any() else None,
        }
    return result


def main():
    # Recent PyTorch ONNX exporters print Unicode status symbols. Windows
    # consoles commonly default to GBK, which otherwise aborts the export.
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(errors="replace")
    parser = argparse.ArgumentParser()
    parser.add_argument("--train", type=Path, required=True)
    parser.add_argument("--test", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--val-fraction", type=float, default=0.2)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--patience", type=int, default=10)
    parser.add_argument("--init-weights", type=Path,
                        help="Initialize from an existing best.pt (optimizer starts fresh)")
    parser.add_argument("--fixed-validation-split", type=Path,
                        help="Existing split_subjects.json; preserve its validation and test subjects")
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--lr-patience", type=int, default=5)
    parser.add_argument("--lr-factor", type=float, default=0.5)
    parser.add_argument("--min-lr", type=float, default=1e-6)
    parser.add_argument("--gpu-preload", action="store_true",
                        help="Keep normalized training windows on CUDA to reduce loader overhead")
    args = parser.parse_args()
    if not 0 < args.val_fraction < 1 or min(args.epochs, args.batch_size, args.patience) < 1:
        parser.error("Require 0 < val-fraction < 1 and positive epochs, batch-size, patience")
    if not (args.lr > 0 and args.min_lr > 0 and args.lr_patience > 0 and
            0 < args.lr_factor < 1):
        parser.error("Require positive learning rates and lr-patience, and 0 < lr-factor < 1")

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    args.out.mkdir(parents=True, exist_ok=True)
    (args.out / "run_config.json").write_text(json.dumps({
        **{k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()},
        "torch_version": torch.__version__, "numpy_version": np.__version__,
        "selection_metric": "validation MSE on BP/100", "optimizer": "Adam",
        "scheduler": "ReduceLROnPlateau", "checkpoint_contains_optimizer": False,
    }, indent=2))
    train_x, train_y, train_subjects = load_subset(args.train)
    test_x, test_y, test_subjects = load_subset(args.test)
    overlap = set(train_subjects) & set(test_subjects)
    if overlap:
        raise ValueError(f"Train/test subject overlap: {len(overlap)} subjects")
    if args.fixed_validation_split is None:
        train_idx, val_idx = split_by_subject(train_subjects, args.val_fraction, args.seed)
    else:
        fixed = json.loads(args.fixed_validation_split.read_text())
        heldout_val = set(map(str, fixed["validation"]))
        heldout_test = set(map(str, fixed["test"]))
        if heldout_test != set(test_subjects) or heldout_val & set(test_subjects):
            raise ValueError("Fixed test subjects differ from the supplied test file")
        if not heldout_val or not heldout_val.issubset(set(train_subjects)):
            raise ValueError("Fixed validation subjects missing from training pool")
        val_mask = np.isin(train_subjects, list(heldout_val))
        train_idx, val_idx = np.flatnonzero(~val_mask), np.flatnonzero(val_mask)
    split_manifest = {
        "seed": args.seed,
        "train": np.unique(train_subjects[train_idx]).tolist(),
        "validation": np.unique(train_subjects[val_idx]).tolist(),
        "test": np.unique(test_subjects).tolist(),
    }
    (args.out / "split_subjects.json").write_text(json.dumps(split_manifest, indent=2))

    # Statistics are computed from training windows only, in float64.
    mean = float(np.mean(train_x[train_idx], dtype=np.float64))
    std = float(np.std(train_x[train_idx], dtype=np.float64))
    if std < 1e-8:
        raise ValueError("PPG training signal is constant")
    preprocess = {"sample_rate_hz": 125, "window_samples": 1250,
                  "mean": mean, "std": std, "output_order": ["sbp", "dbp"],
                  "onnx_output": "bp_half", "output_offset_mmhg": [0.0, 0.0],
                  "output_scale_mmhg": 2.0}
    (args.out / "preprocess.json").write_text(json.dumps(preprocess, indent=2))
    train_loader = DataLoader(Windows(train_x, train_y, train_idx, mean, std),
                              batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(Windows(train_x, train_y, val_idx, mean, std),
                            batch_size=args.batch_size)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if args.gpu_preload and device.type != "cuda":
        parser.error("--gpu-preload requires CUDA-enabled PyTorch and a usable GPU")
    if args.gpu_preload:
        gpu_x = torch.from_numpy(((train_x[train_idx] - mean) / std)
                                 .astype(np.float32)[:, None, :]).to(device)
        gpu_y = torch.from_numpy(train_y[train_idx].copy()).to(device)
    model = TinyBP().to(device)
    if args.init_weights is not None:
        model.load_state_dict(torch.load(args.init_weights, map_location=device,
                                         weights_only=True))
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=args.lr_factor, patience=args.lr_patience,
        min_lr=args.min_lr)
    best_loss = float("inf")
    stale_epochs = 0
    history = []
    writer = SummaryWriter(log_dir=str(args.out / "tensorboard"))
    writer.add_histogram("labels/train_sbp_mmhg", train_y[train_idx, 0], 0)
    writer.add_histogram("labels/train_dbp_mmhg", train_y[train_idx, 1], 0)
    writer.add_histogram("labels/test_sbp_mmhg", test_y[:, 0], 0)
    writer.add_histogram("labels/test_dbp_mmhg", test_y[:, 1], 0)
    writer.add_text("run/data", f"train={args.train}; test={args.test}; seed={args.seed}", 0)
    for epoch in range(args.epochs):
        model.train()
        train_sum, train_n = 0.0, 0
        if args.gpu_preload:
            order = torch.randperm(len(gpu_x), device=device)
            for batch_ids in order.split(args.batch_size):
                x, y = gpu_x[batch_ids], gpu_y[batch_ids]
                optimizer.zero_grad()
                loss = torch.nn.functional.mse_loss(model(x) / 100.0, y / 100.0)
                loss.backward()
                optimizer.step()
                train_sum += loss.item() * len(x)
                train_n += len(x)
        else:
            for x, y in train_loader:
                x, y = x.to(device), y.to(device)
                optimizer.zero_grad()
                loss = torch.nn.functional.mse_loss(model(x) / 100.0, y / 100.0)
                loss.backward()
                optimizer.step()
                train_sum += loss.item() * len(x)
                train_n += len(x)
        model.eval()
        with torch.no_grad():
            val_sum, val_n = 0.0, 0
            for x, y in val_loader:
                batch_loss = torch.nn.functional.mse_loss(model(x.to(device)) / 100.0,
                                                          y.to(device) / 100.0).item()
                val_sum += batch_loss * len(x)
                val_n += len(x)
            val_loss = val_sum / val_n
        train_loss = train_sum / train_n
        val_metrics = score(model, val_loader, device)
        current_lr = optimizer.param_groups[0]["lr"]
        writer.add_scalar("loss/train_mse_scaled", train_loss, epoch + 1)
        writer.add_scalar("loss/val_mse_scaled", val_loss, epoch + 1)
        writer.add_scalar("train/learning_rate", current_lr, epoch + 1)
        writer.add_scalar("mae/val_sbp_mmhg", val_metrics["mae_sbp"], epoch + 1)
        writer.add_scalar("mae/val_dbp_mmhg", val_metrics["mae_dbp"], epoch + 1)
        writer.flush()
        history.append({"epoch": epoch + 1, "train_mse_scaled": train_loss,
                        "val_mse_scaled": val_loss,
                        "learning_rate": current_lr,
                        "val_sbp_mae": val_metrics["mae_sbp"],
                        "val_dbp_mae": val_metrics["mae_dbp"]})
        print(f"epoch {epoch + 1}: train={train_loss:.5f}, val={val_loss:.5f}, "
              f"SBP MAE={val_metrics['mae_sbp']:.2f}, DBP MAE={val_metrics['mae_dbp']:.2f}, "
              f"lr={current_lr:.7f}",
              flush=True)
        scheduler.step(val_loss)
        if val_loss < best_loss:
            best_loss = val_loss
            torch.save(model.cpu().state_dict(), args.out / "best.pt")
            model.to(device)
            stale_epochs = 0
        else:
            stale_epochs += 1
            if stale_epochs >= args.patience:
                print(f"Early stopping after {epoch + 1} epochs", flush=True)
                break

    (args.out / "history.json").write_text(json.dumps(history, indent=2))

    model.cpu().load_state_dict(torch.load(args.out / "best.pt", map_location="cpu",
                                          weights_only=True))
    val_metrics = score(model, val_loader, torch.device("cpu"))
    test_loader = DataLoader(Windows(test_x, test_y, np.arange(len(test_x)), mean, std),
                             batch_size=args.batch_size)
    test_metrics = score(model, test_loader, torch.device("cpu"))
    metrics = {"validation": val_metrics, "test": test_metrics,
               "train_subjects": len(np.unique(train_subjects[train_idx])),
               "validation_subjects": len(np.unique(train_subjects[val_idx])),
               "test_subjects": len(np.unique(test_subjects)),
               "parameters": sum(p.numel() for p in model.parameters())}
    baseline = train_y[train_idx].mean(axis=0)
    metrics["constant_mean_baseline_test_mae"] = {
        "sbp": float(np.abs(test_y[:, 0] - baseline[0]).mean()),
        "dbp": float(np.abs(test_y[:, 1] - baseline[1]).mean()),
    }
    metrics["within_case_change_test"] = trend_score(
        model, args.test, test_x, test_y, mean, std, args.batch_size)
    writer.add_scalar("final/test_sbp_mae_mmhg", test_metrics["mae_sbp"], len(history))
    writer.add_scalar("final/test_dbp_mae_mmhg", test_metrics["mae_dbp"], len(history))
    writer.add_scalar("final/constant_sbp_mae_mmhg",
                      metrics["constant_mean_baseline_test_mae"]["sbp"], len(history))
    writer.add_scalar("final/constant_dbp_mae_mmhg",
                      metrics["constant_mean_baseline_test_mae"]["dbp"], len(history))
    writer.close()
    (args.out / "metrics.json").write_text(json.dumps(metrics, indent=2))
    export_model = TinyBPExport(model.eval()).eval()
    torch.onnx.export(export_model, torch.zeros(1, 1, 1250), args.out / "tiny_bp.onnx",
                      input_names=["ppg"], output_names=["bp_half"], opset_version=18)
    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()
