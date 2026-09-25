"""Re-export an existing trained TinyBP checkpoint without retraining."""

import argparse
import json
import sys
from pathlib import Path

import torch

from model import TinyBP, TinyBPExport


def main():
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(errors="replace")
    parser = argparse.ArgumentParser()
    parser.add_argument("--run", type=Path, required=True)
    args = parser.parse_args()
    model = TinyBP()
    model.load_state_dict(torch.load(args.run / "best.pt", map_location="cpu",
                                     weights_only=True))
    model.eval()
    preprocess_path = args.run / "preprocess.json"
    preprocess = json.loads(preprocess_path.read_text())
    preprocess.update({"onnx_output": "bp_half", "output_offset_mmhg": [0.0, 0.0],
                       "output_scale_mmhg": 2.0})
    preprocess_path.write_text(json.dumps(preprocess, indent=2))
    torch.onnx.export(TinyBPExport(model).eval(), torch.zeros(1, 1, 1250),
                      args.run / "tiny_bp.onnx", input_names=["ppg"],
                      output_names=["bp_half"], opset_version=18)


if __name__ == "__main__":
    main()
