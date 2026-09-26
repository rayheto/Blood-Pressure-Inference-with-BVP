"""Smoke-check 10-second streaming inference on an original VitalDB recording."""

import argparse
import json
from pathlib import Path

import numpy as np
from scipy.signal import find_peaks

from build_ecg_windows import get_track_map, load_wave
from stream_inference import StreamingBloodPressure
from train_delta import score


def arterial_label(art):
    if not np.isfinite(art).all() or np.mean((art >= 30) & (art <= 250)) < .99:
        return None
    peaks, _ = find_peaks(art, distance=40, prominence=8)
    feet, _ = find_peaks(-art, distance=40, prominence=8)
    if len(peaks) < 4 or len(feet) < 4:
        return None
    sbp, dbp = np.median(art[peaks]), np.median(art[feet])
    if not (60 <= sbp <= 240 and 30 <= dbp <= 150 and 15 <= sbp - dbp <= 130):
        return None
    return np.array([sbp, dbp], np.float32)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--caseid", required=True)
    p.add_argument("--tracks", type=Path, required=True)
    p.add_argument("--preprocess", type=Path, required=True)
    p.add_argument("--run", type=Path, required=True)
    p.add_argument("--trace", type=Path, help="Write per-window reference and predictions for plotting")
    p.add_argument("--heads", type=Path, help="Directory containing trained correction heads")
    p.add_argument("--report", type=Path, help="Override report JSON path")
    args = p.parse_args()
    tracks = get_track_map(args.tracks)
    names = ("SNUADC/PLETH", "SNUADC/ECG_II", "SNUADC/ART")
    ids = [tracks.get((args.caseid, name)) for name in names]
    if any(tid is None for tid in ids):
        raise ValueError("Missing synchronized PPG, ECG, or ART track")
    ppg, ecg, art = (load_wave(tid)[::4] for tid in ids)
    length = min(len(ppg), len(ecg), len(art))
    models = {name: StreamingBloodPressure(args.run / f"{name}.pt", args.preprocess, mode=name)
              for name in ("ppg_only", "ppg_pat_rr")}
    if args.heads:
        for name in ("uniform", "high_pressure_weighted"):
            models[name] = StreamingBloodPressure(args.run / "ppg_pat_rr.pt", args.preprocess,
                                                  mode="ppg_pat_rr",
                                                  correction_checkpoint=args.heads / f"{name}.pt")
    truth, preds, ages = [], {name: [] for name in models}, []
    first_time, first_bp = None, None
    for start in range(0, length - 1249, 1250):
        end_s = (start + 1250) / 125
        pw, ew, aw = ppg[start:start + 1250], ecg[start:start + 1250], art[start:start + 1250]
        label = arterial_label(aw)
        if label is None or not np.isfinite(pw).all() or not np.isfinite(ew).all():
            continue
        if np.std(pw) < 1e-4 or np.std(ew) < .015:
            continue
        if first_time is None:
            if end_s < 120:
                continue
            first_time, first_bp = end_s, label
            for model in models.values():
                model.calibrate(pw, ew, float(label[0]), float(label[1]), end_s)
            continue
        age = end_s - first_time
        if age > 1800:
            break
        outputs = {name: model.observe_window(pw, ew, end_s) for name, model in models.items()}
        if any(value["status"] != "estimate" for value in outputs.values()):
            continue
        truth.append(label - first_bp)
        for name, value in outputs.items():
            preds[name].append([value["delta_sbp_mmhg"], value["delta_dbp_mmhg"]])
        ages.append(age)
    if not truth:
        raise ValueError("No valid consecutive windows after calibration")
    truth = np.asarray(truth)
    report = {"caseid": args.caseid, "source": "original continuous VitalDB 500-Hz tracks downsampled to 125 Hz",
              "calibration_time_s": first_time, "evaluated_10s_windows": len(truth),
              "elapsed_last_s": ages[-1], "reference": "ART used to emulate cuff for research",
              "zero_change": score(np.zeros_like(truth), truth),
              "models": {name: score(np.asarray(values), truth) for name, values in preds.items()}}
    output = args.report or args.run / f"raw_stream_case_{args.caseid}.json"
    output.write_text(json.dumps(report, indent=2))
    if args.trace:
        trace = {
            "caseid": args.caseid,
            "calibration_time_s": first_time,
            "calibration_bp_mmhg": first_bp.tolist(),
            "elapsed_s": ages,
            "reference_bp_mmhg": (truth + first_bp).tolist(),
            **{name + "_bp_mmhg": (np.asarray(values) + first_bp).tolist()
               for name, values in preds.items()},
        }
        args.trace.parent.mkdir(parents=True, exist_ok=True)
        args.trace.write_text(json.dumps(trace, separators=(",", ":")))
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
