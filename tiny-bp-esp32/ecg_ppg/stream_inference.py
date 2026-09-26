"""Causal 10-second ECG/PPG streaming adapter for the VitalDB change model."""

import json
from collections import deque
from pathlib import Path

import numpy as np
import torch
from scipy.signal import butter

from train_delta import DeltaBP
from train_fusion import signal_features
from train_pat_delta import pair_features
from train_dynamic_head import DynamicHead, inputs as dynamic_inputs


WINDOW_SAMPLES = 1250
SAMPLE_RATE_HZ = 125
MAX_CALIBRATION_AGE_S = 1800


class StreamingBloodPressure:
    """Returns a new estimate after each completed 10-second signal window.

    The reference SBP/DBP must come from an external cuff at calibration time.
    No arterial-pressure label or later cuff value is used by `observe_window`.
    """

    def __init__(self, checkpoint: Path, preprocess: Path, device="cpu", mode="ppg_pat_rr",
                 correction_checkpoint=None):
        if mode not in ("ppg_only", "ppg_pat_rr"):
            raise ValueError("mode must be ppg_only or ppg_pat_rr")
        self.mode = mode
        self.device = torch.device(device)
        config = json.loads(Path(preprocess).read_text())
        self.ppg_mean = float(config["mean"])
        self.ppg_std = float(config["std"])
        if not np.isfinite(self.ppg_std) or self.ppg_std <= 0:
            raise ValueError("Invalid PPG normalization")
        self.model = DeltaBP(1, 4 if mode == "ppg_pat_rr" else 0).to(self.device).eval()
        self.model.load_state_dict(torch.load(checkpoint, map_location=self.device, weights_only=True))
        self.correction = None
        if correction_checkpoint is not None:
            if mode != "ppg_pat_rr":
                raise ValueError("Dynamic correction requires ppg_pat_rr mode")
            self.correction = DynamicHead().to(self.device).eval()
            self.correction.load_state_dict(torch.load(correction_checkpoint, map_location=self.device, weights_only=True))
        self.ecg_filter = butter(2, (5, 20), btype="bandpass", fs=SAMPLE_RATE_HZ, output="sos")
        self.calibration = None
        self.ppg_buffer = deque()
        self.ecg_buffer = deque()

    def _window(self, ppg, ecg):
        ppg = np.asarray(ppg, dtype=np.float32)
        ecg = np.zeros_like(ppg) if ecg is None and self.mode == "ppg_only" else np.asarray(ecg, dtype=np.float32)
        if ppg.shape != (WINDOW_SAMPLES,) or ecg.shape != (WINDOW_SAMPLES,):
            raise ValueError("Expected 1,250-sample PPG and ECG windows")
        if not np.isfinite(ppg).all() or not np.isfinite(ecg).all():
            raise ValueError("Input contains nonfinite samples")
        if np.std(ppg) < 1e-4:
            raise ValueError("PPG window is nearly flat")
        x = ((ppg - self.ppg_mean) / self.ppg_std).astype(np.float32)[None, None, :]
        timing = (signal_features(ppg, ecg, self.ecg_filter) if self.mode == "ppg_pat_rr"
                  else np.zeros(3, np.float32))
        return x, timing

    def calibrate(self, ppg, ecg, sbp, dbp, window_end_s):
        """Save one completed signal window and a simultaneous cuff BP reading."""
        if not (50 <= sbp <= 260 and 30 <= dbp <= 160 and sbp > dbp):
            raise ValueError("Calibration blood pressure is outside expected range")
        x, timing = self._window(ppg, ecg)
        self.calibration = (x, timing, np.array([sbp, dbp], np.float32), float(window_end_s))
        return {"status": "calibrated", "window_end_s": float(window_end_s)}

    def observe_window(self, ppg, ecg, window_end_s):
        """Predict from a *completed* 10-second window; no future samples."""
        if self.calibration is None:
            return {"status": "calibration_required"}
        x0, timing0, cuff, time0 = self.calibration
        age = float(window_end_s) - time0
        if age < 10:
            return {"status": "waiting_for_next_window", "elapsed_s": age}
        if age > MAX_CALIBRATION_AGE_S:
            return {"status": "recalibration_required", "elapsed_s": age}
        x1, timing1 = self._window(ppg, ecg)
        if self.mode == "ppg_pat_rr":
            extra, pat_valid = pair_features(np.stack((timing0, timing1)),
                                             np.array([[0, 1]], np.int32), True)
            extra_tensor = torch.from_numpy(extra).to(self.device)
        else:
            pat_valid = np.array([False])
            extra_tensor = None
        with torch.no_grad():
            delta = self.model(torch.from_numpy(x0).to(self.device),
                               torch.from_numpy(x1).to(self.device),
                               torch.tensor([[age / MAX_CALIBRATION_AGE_S]], dtype=torch.float32, device=self.device),
                               extra_tensor).cpu().numpy()[0]
            base_delta = delta.copy()
            if self.correction is not None:
                state = dynamic_inputs(delta[None], cuff[None], np.array([age], np.float32))
                delta += self.correction(torch.from_numpy(state).to(self.device)).cpu().numpy()[0]
        estimate = cuff + delta
        return {"status": "estimate", "window_end_s": float(window_end_s),
                "elapsed_s": age, "sbp_mmhg": float(estimate[0]),
                "dbp_mmhg": float(estimate[1]),
                "delta_sbp_mmhg": float(delta[0]),
                "delta_dbp_mmhg": float(delta[1]),
                "base_delta_sbp_mmhg": float(base_delta[0]),
                "base_delta_dbp_mmhg": float(base_delta[1]),
                "pat_detected_in_both_windows": bool(pat_valid[0])}

    def push_samples(self, ppg_samples, ecg_samples, last_sample_time_s):
        """Accept equal-length, ordered 125-Hz chunks; emit each full window."""
        ppg = np.asarray(ppg_samples, dtype=np.float32).reshape(-1)
        ecg = (np.zeros_like(ppg) if ecg_samples is None and self.mode == "ppg_only"
               else np.asarray(ecg_samples, dtype=np.float32).reshape(-1))
        if len(ppg) != len(ecg):
            raise ValueError("PPG and ECG chunks must contain the same number of samples")
        self.ppg_buffer.extend(ppg.tolist())
        self.ecg_buffer.extend(ecg.tolist())
        outputs = []
        while len(self.ppg_buffer) >= WINDOW_SAMPLES:
            remaining = len(self.ppg_buffer) - WINDOW_SAMPLES
            end_s = float(last_sample_time_s) - remaining / SAMPLE_RATE_HZ
            pw = np.fromiter((self.ppg_buffer.popleft() for _ in range(WINDOW_SAMPLES)), np.float32)
            ew = np.fromiter((self.ecg_buffer.popleft() for _ in range(WINDOW_SAMPLES)), np.float32)
            outputs.append(self.observe_window(pw, ew, end_s))
        return outputs
