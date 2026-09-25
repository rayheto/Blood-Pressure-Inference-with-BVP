"""Convert ONNX to ESP-DL int8 using representative PPG calibration windows."""

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset

from esp_ppq.api import espdl_quantize_onnx
from esp_ppq.api import QuantizationSettingFactory
from esp_ppq.executor import TorchExecutor


def main():
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(errors="replace")
    parser = argparse.ArgumentParser()
    parser.add_argument("--onnx", type=Path, required=True)
    parser.add_argument("--calib", type=Path, required=True,
                        help="NPZ with real ppg_signals (N, 1250) from target sensor")
    parser.add_argument("--preprocess", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--steps", type=int, default=32)
    parser.add_argument("--activation-calibration", choices=("kl", "minmax"),
                        default="kl", help="Use minmax when output clipping matters")
    parser.add_argument("--eval-npz", type=Path,
                        help="Independent labeled NPZ for quantization impact")
    parser.add_argument("--checkpoint", type=Path,
                        help="PyTorch checkpoint for float-reference comparison")
    args = parser.parse_args()
    if args.steps < 1:
        parser.error("--steps must be at least 1")
    preprocess = json.loads(args.preprocess.read_text())
    with np.load(args.calib, allow_pickle=False) as data:
        signals = np.asarray(data["ppg_signals"], dtype=np.float32)
    if signals.ndim != 2 or signals.shape[1] != 1250:
        raise ValueError(f"Expected (N, 1250) calibration PPG, got {signals.shape}")
    if len(signals) == 0 or not np.isfinite(signals).all():
        raise ValueError("Calibration PPG must be nonempty and finite")
    # Input files are grouped by case. Spread calibration windows over the
    # entire file so they represent multiple subjects instead of case one.
    selected = np.linspace(0, len(signals) - 1, min(args.steps, len(signals)),
                           dtype=np.int64)
    signals = signals[selected]
    normalized = ((signals - preprocess["mean"]) / preprocess["std"]).astype(np.float32)
    x = torch.from_numpy(normalized[:, None, :])
    loader = DataLoader(TensorDataset(x), batch_size=1, shuffle=False)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    setting = QuantizationSettingFactory.espdl_setting()
    setting.quantize_activation_setting.calib_algorithm = args.activation_calibration
    graph = espdl_quantize_onnx(
        onnx_import_file=str(args.onnx),
        espdl_export_file=str(args.out),
        calib_dataloader=loader,
        calib_steps=len(loader),
        input_shape=[1, 1, 1250],
        target="esp32s3",
        quant_type="w8a8",
        setting=setting,
        collate_fn=lambda batch: batch[0],
        device="cpu",
        error_report=True,
        export_test_values=True,
        verbose=0,
    )
    if not args.out.is_file():
        raise RuntimeError("ESP-DL conversion returned without an output model")
    print(f"Wrote {args.out} ({args.out.stat().st_size} bytes)")
    if args.eval_npz is not None:
        if args.checkpoint is None:
            parser.error("--eval-npz also requires --checkpoint")
        from model import TinyBP
        float_model = TinyBP().eval()
        float_model.load_state_dict(torch.load(args.checkpoint, map_location="cpu",
                                               weights_only=True))
        executor = TorchExecutor(graph=graph, device="cpu")
        with np.load(args.eval_npz, allow_pickle=False) as data:
            eval_signals = np.asarray(data["ppg_signals"], dtype=np.float32)
            targets = np.stack((data["sbp"], data["dbp"]), axis=1).astype(np.float32)
        float_out, quant_out = [], []
        with torch.no_grad():
            for start in range(0, len(eval_signals), 64):
                end = min(start + 64, len(eval_signals))
                batch = torch.from_numpy(((eval_signals[start:end] - preprocess["mean"])
                                          / preprocess["std"])[:, None, :].astype(np.float32))
                float_out.append(float_model(batch).numpy())
                # ESP-DL runs batch size 1; use the same size in simulation.
                for sample in batch:
                    result = executor(sample[None])[0]
                    quant_out.append(result.detach().numpy()[0] *
                                     preprocess["output_scale_mmhg"])
        float_out = np.concatenate(float_out)
        quant_out = np.asarray(quant_out)
        report = {
            "windows": len(targets),
            "float_mae_mmhg": np.mean(np.abs(float_out - targets), axis=0).tolist(),
            "quantized_mae_mmhg": np.mean(np.abs(quant_out - targets), axis=0).tolist(),
            "quantized_minus_float_mae_mmhg":
                (np.mean(np.abs(quant_out - targets), axis=0) -
                 np.mean(np.abs(float_out - targets), axis=0)).tolist(),
            "quantized_vs_float_mae_mmhg":
                np.mean(np.abs(quant_out - float_out), axis=0).tolist(),
            "quantized_vs_float_max_abs_mmhg":
                np.max(np.abs(quant_out - float_out), axis=0).tolist(),
        }
        report_path = args.out.with_name(args.out.stem + "_eval.json")
        report_path.write_text(json.dumps(report, indent=2))
        print(f"Wrote {report_path}")


if __name__ == "__main__":
    main()
