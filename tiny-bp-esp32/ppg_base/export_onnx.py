"""Export the selected PPG base and correction head as one ONNX graph."""
import argparse
import sys
from pathlib import Path

import numpy as np
import torch
from torch import nn

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from model import TinyBP  # noqa: E402


class CurrentPpg(nn.Module):
    def __init__(self, base, head):
        super().__init__()
        self.base = base
        self.register_buffer("mean", torch.from_numpy(head["mean"].astype(np.float32)))
        self.register_buffer("std", torch.from_numpy(head["std"].astype(np.float32)))
        self.fc0 = nn.Linear(35, 32)
        self.fc1 = nn.Linear(32, 2)
        with torch.no_grad():
            self.fc0.weight.copy_(torch.from_numpy(head["fc0_weight"]))
            self.fc0.bias.copy_(torch.from_numpy(head["fc0_bias"]))
            self.fc1.weight.copy_(torch.from_numpy(head["fc1_weight"]))
            self.fc1.bias.copy_(torch.from_numpy(head["fc1_bias"]))

    def forward(self, ppg):
        latent = self.base.pool(self.base.features(ppg)).flatten(1)
        tiny = self.base.head(latent) * 100.0
        current = torch.cat((latent, tiny / 100.0), dim=1)
        valid = torch.ones_like(current[:, :1])
        z = torch.cat(((current - self.mean) / self.std, valid), dim=1)
        corrected = tiny + torch.tanh(self.fc1(torch.relu(self.fc0(z)))) * ppg.new_tensor([40.0, 25.0])
        return corrected / 4.0  # preserve dynamic range for ESP-DL quantization


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", type=Path, default=ROOT / "ppg_base/model.onnx")
    args = parser.parse_args()
    base = TinyBP().eval()
    base.load_state_dict(torch.load(ROOT / "model/best.pt", map_location="cpu", weights_only=True))
    with np.load(ROOT / "ppg_base/selected_head.npz", allow_pickle=False) as source:
        head = {key: source[key] for key in source.files}
    model = CurrentPpg(base, head).eval()
    args.out.parent.mkdir(parents=True, exist_ok=True)
    torch.onnx.export(model, torch.zeros(1, 1, 1250), args.out,
                      input_names=["ppg_normalized"], output_names=["bp_quarter_mmhg"],
                      opset_version=18, dynamo=False)
    print(args.out)


if __name__ == "__main__":
    main()
