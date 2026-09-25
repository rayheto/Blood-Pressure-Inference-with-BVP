"""Small convolutional research baseline for 1250-sample PPG windows."""

import torch
from torch import nn


class TinyBP(nn.Module):
    def __init__(self):
        super().__init__()
        layers = []
        channels = (1, 8, 16, 24, 32)
        for input_channels, output_channels in zip(channels[:-1], channels[1:]):
            layers.extend((
                nn.Conv1d(input_channels, output_channels, kernel_size=9,
                          stride=2, padding=4, bias=False),
                nn.BatchNorm1d(output_channels),
                nn.ReLU(),
            ))
        self.features = nn.Sequential(*layers)
        self.pool = nn.AdaptiveAvgPool1d(1)
        self.head = nn.Linear(32, 2)
        # Start near a population-scale BP estimate so short training runs do
        # not waste many epochs moving the outputs up from 0 mmHg.
        with torch.no_grad():
            self.head.bias.copy_(torch.tensor([1.2, 0.8]))

    def forward(self, ppg):
        # Predict SBP and DBP in mmHg. Training uses targets/100 for stability.
        x = self.pool(self.features(ppg)).flatten(1)
        return self.head(x) * 100.0


class TinyBPExport(nn.Module):
    """Scale hardware output to avoid int8 saturation in mmHg."""

    def __init__(self, trained_model):
        super().__init__()
        self.trained_model = trained_model

    def forward(self, ppg):
        return self.trained_model(ppg) / 2.0
