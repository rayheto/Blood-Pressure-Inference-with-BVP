"""Train repository architectures and evaluate them alongside TinyBP on one VitalDB split.

The repository's published PulseDB checkpoints are not available here. These
baselines are newly trained on the same VitalDB train/validation subjects.
"""

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src.dl_models import ResNet1D, ResNetBiGRU  # noqa: E402
from model import TinyBP  # noqa: E402


def metrics(prediction, label):
    error = prediction - label
    return {
        "mae_sbp": float(np.abs(error[:, 0]).mean()),
        "mae_dbp": float(np.abs(error[:, 1]).mean()),
        "bias_sbp": float(error[:, 0].mean()),
        "bias_dbp": float(error[:, 1].mean()),
        "sd_sbp": float(error[:, 0].std(ddof=1)),
        "sd_dbp": float(error[:, 1].std(ddof=1)),
        "windows": len(error),
    }


def predict(model, x, batch_size, device, scale=1.0):
    model.eval()
    chunks = []
    with torch.no_grad():
        for start in range(0, len(x), batch_size):
            with torch.autocast(device_type="cuda", enabled=device.type == "cuda"):
                output = model(x[start:start + batch_size].to(device))
            chunks.append(output.float().cpu().numpy() * scale)
    return np.concatenate(chunks)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--train", type=Path, required=True)
    parser.add_argument("--test", type=Path, required=True)
    parser.add_argument("--split", type=Path, required=True)
    parser.add_argument("--preprocess", type=Path, default=Path(__file__).parent / "model/preprocess.json")
    parser.add_argument("--tiny-weights", type=Path, default=Path(__file__).parent / "model/best.pt")
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--architecture", choices=("resnet1d", "resnet_bigru"), required=True)
    parser.add_argument("--epochs", type=int, default=12)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--lr", type=float, default=0.001)
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    torch.manual_seed(42)
    np.random.seed(42)
    torch.set_num_threads(4)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    config = json.loads(args.preprocess.read_text())
    split = json.loads(args.split.read_text())
    with np.load(args.train, allow_pickle=False) as data:
        subjects = data["subjects"].astype(str)
        train_ids = np.flatnonzero(np.isin(subjects, split["train"]))
        val_ids = np.flatnonzero(np.isin(subjects, split["validation"]))
        x_pool = torch.from_numpy(((data["ppg_signals"] - config["mean"]) /
                                    config["std"]).astype(np.float32)[:, None, :])
        y_pool = np.stack((data["sbp"], data["dbp"]), axis=1).astype(np.float32)
    with np.load(args.test, allow_pickle=False) as data:
        test_subjects = data["subjects"].astype(str)
        x_test = torch.from_numpy(((data["ppg_signals"] - config["mean"]) /
                                    config["std"]).astype(np.float32)[:, None, :])
        y_test = np.stack((data["sbp"], data["dbp"]), axis=1).astype(np.float32)
    assert len(train_ids) == 307079 and len(val_ids) == 2600 and len(x_test) == 3200
    assert set(subjects[train_ids]).isdisjoint(subjects[val_ids])
    assert set(subjects).isdisjoint(test_subjects)
    assert set(test_subjects) == set(split["test"])
    del subjects, test_subjects

    tiny = TinyBP().to(device)
    tiny.load_state_dict(torch.load(args.tiny_weights, map_location=device, weights_only=True))
    tiny_scores = metrics(predict(tiny, x_test, args.batch_size, device), y_test)
    print("tiny", json.dumps(tiny_scores), flush=True)
    del tiny

    architecture = ResNet1D if args.architecture == "resnet1d" else ResNetBiGRU
    predictions = np.zeros_like(y_test)
    chosen_epochs = {}
    for target, column in (("sbp", 0), ("dbp", 1)):
        torch.manual_seed(42 + column)
        model = architecture().to(device)
        optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=1e-5)
        scaler = torch.amp.GradScaler("cuda", enabled=device.type == "cuda")
        y_train = torch.from_numpy(y_pool[train_ids, column] / 100.0)
        y_val = torch.from_numpy(y_pool[val_ids, column] / 100.0)
        best_loss = float("inf")
        best_state = None
        x_val = x_pool[val_ids]
        for epoch in range(1, args.epochs + 1):
            started = time.time()
            model.train()
            order = torch.randperm(len(train_ids))
            running = 0.0
            for batch in order.split(args.batch_size):
                x = x_pool[train_ids[batch.numpy()]].to(device)
                y = y_train[batch].to(device)
                optimizer.zero_grad(set_to_none=True)
                with torch.autocast(device_type="cuda", enabled=device.type == "cuda"):
                    loss = torch.nn.functional.mse_loss(model(x).flatten(), y)
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
                running += float(loss.detach()) * len(batch)
            val_pred = predict(model, x_val, args.batch_size, device).flatten()
            val_loss = float(np.mean((val_pred - y_val.numpy()) ** 2))
            if val_loss < best_loss:
                best_loss = val_loss
                best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
                chosen_epochs[target] = epoch
            print(f"{args.architecture} {target} epoch {epoch}/{args.epochs} "
                  f"train_mse={running / len(train_ids):.5f} val_mse={val_loss:.5f} "
                  f"seconds={time.time() - started:.1f}", flush=True)
        model.load_state_dict(best_state)
        torch.save(best_state, args.out / f"{args.architecture}_{target}.pt")
        predictions[:, column] = predict(model, x_test, args.batch_size, device, scale=100).flatten()
        del model, optimizer, scaler, y_train, y_val, x_val, best_state
        torch.cuda.empty_cache()

    report = {
        "dataset": "VitalDB direct PLETH/ART, same 32 test subjects and 3200 windows as TinyBP",
        "architecture": args.architecture,
        "training_subjects": 3075,
        "training_windows": len(train_ids),
        "validation_subjects": 26,
        "validation_windows": len(val_ids),
        "test_subjects": 32,
        "epochs_requested": args.epochs,
        "best_epoch": chosen_epochs,
        "baseline_metrics": metrics(predictions, y_test),
        "tiny_metrics_same_run": tiny_scores,
        "note": "Repository architecture freshly trained here; not its published PulseDB weights.",
    }
    (args.out / f"{args.architecture}_comparison.json").write_text(json.dumps(report, indent=2))
    print(json.dumps(report, indent=2), flush=True)


if __name__ == "__main__":
    main()
