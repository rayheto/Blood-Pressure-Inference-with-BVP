"""Quantize the complete current-PPG ONNX graph for ESP32-S3."""
import argparse
import json
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset
from esp_ppq.api import QuantizationSettingFactory, espdl_quantize_onnx
from esp_ppq.executor import TorchExecutor

ROOT = Path(__file__).resolve().parents[1]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--onnx", type=Path, default=ROOT / "ppg_base/model.onnx")
    parser.add_argument("--calib", type=Path, required=True,
                        help="NPZ with raw ppg_signals of shape (N,1250), separate from test cases")
    parser.add_argument("--out", type=Path, default=ROOT / "firmware/main/models/s3/ppg_base.espdl")
    parser.add_argument("--quant-type", choices=("w8a8", "w8a16", "w16a16"), default="w8a16")
    parser.add_argument("--eval", type=Path, help="Independent raw PPG NPZ for float/quantized comparison")
    args = parser.parse_args()
    preprocess = json.loads((ROOT / "model/preprocess.json").read_text())
    with np.load(args.calib, allow_pickle=False) as source:
        raw = np.asarray(source["ppg_signals"], dtype=np.float32)
    if raw.ndim != 2 or raw.shape[1] != 1250 or not np.isfinite(raw).all():
        raise ValueError("Calibration input must be finite (N,1250) raw PPG")
    samples = torch.from_numpy(((raw - preprocess["mean"]) / preprocess["std"])[:, None, :])
    loader = DataLoader(TensorDataset(samples), batch_size=1, shuffle=False)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    setting = QuantizationSettingFactory.espdl_setting()
    setting.quantize_activation_setting.calib_algorithm = "minmax"
    graph = espdl_quantize_onnx(
        onnx_import_file=str(args.onnx), espdl_export_file=str(args.out),
        calib_dataloader=loader, calib_steps=len(loader), input_shape=[1, 1, 1250],
        target="esp32s3", quant_type=args.quant_type, setting=setting,
        collate_fn=lambda batch: batch[0], device="cpu", error_report=True,
        export_test_values=True, verbose=0,
    )
    print(f"Wrote {args.out} ({args.out.stat().st_size} bytes)")
    if args.eval:
        import onnxruntime as ort
        with np.load(args.eval, allow_pickle=False) as source:
            eval_raw = np.asarray(source["ppg_signals"], dtype=np.float32)
        eval_x = ((eval_raw - preprocess["mean"]) / preprocess["std"])[:, None, :].astype(np.float32)
        session = ort.InferenceSession(str(args.onnx), providers=["CPUExecutionProvider"])
        executor = TorchExecutor(graph=graph, device="cpu")
        float_bp = np.concatenate([
            session.run(None, {session.get_inputs()[0].name: x[None]})[0]
            for x in eval_x]) * 4.0
        quant_bp = np.stack([executor(torch.from_numpy(x[None]))[0].detach().numpy()[0]
                             for x in eval_x]) * 4.0
        delta = np.abs(quant_bp - float_bp)
        result = {"windows": len(eval_x), "quant_type": args.quant_type,
                  "output_scale_mmhg": 4.0,
                  "first_float_mmhg": float_bp[0].tolist(),
                  "first_quantized_mmhg": quant_bp[0].tolist(),
                  "mean_abs_diff_mmhg": delta.mean(axis=0).tolist(),
                  "max_abs_diff_mmhg": delta.max(axis=0).tolist()}
        report = args.out.with_suffix(".eval.json")
        report.write_text(json.dumps(result, indent=2) + "\n")
        print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
