"""Embed one real VitalDB training window and its matching preprocessing constants."""

import argparse
import hashlib
import json
import sys
from pathlib import Path

import numpy as np
import torch


PROJECT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT))
from model import TinyBP  # noqa: E402


def c_float(value):
    result = format(float(value), ".9g")
    if "." not in result and "e" not in result:
        result += ".0"
    return result + "f"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--train", type=Path, required=True)
    parser.add_argument("--split", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--preprocess", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    config = json.loads(args.preprocess.read_text())
    split = json.loads(args.split.read_text())
    held_out = set(map(str, split["validation"] + split["test"]))
    with np.load(args.train, allow_pickle=False) as data:
        subjects = np.asarray(data["subjects"]).astype(str)
        indices = np.flatnonzero(~np.isin(subjects, list(held_out)))
        if len(indices) == 0:
            raise RuntimeError("No training subject available for fixture")
        sbp = np.asarray(data["sbp"])
        dbp = np.asarray(data["dbp"])
        # Deterministic representative window near a normal BP value.
        score = np.abs(sbp[indices] - 120) + np.abs(dbp[indices] - 80)
        index = int(indices[np.argmin(score)])
        raw = np.ascontiguousarray(data["ppg_signals"][index], dtype=np.float32)
        caseid = str(data["caseids"][index])
        subject = str(data["subjects"][index])
        time_s = int(data["time_s"][index])
        reference = [float(sbp[index]), float(dbp[index])]
    if raw.shape != (config["window_samples"],) or not np.isfinite(raw).all():
        raise ValueError("Invalid fixture waveform")

    model = TinyBP().eval()
    model.load_state_dict(torch.load(args.checkpoint, map_location="cpu", weights_only=True))
    normalized = ((raw - config["mean"]) / config["std"]).astype(np.float32)
    with torch.no_grad():
        prediction = model(torch.from_numpy(normalized[None, None, :])).numpy()[0]

    values = [c_float(v) for v in raw]
    rows = ["    " + ", ".join(values[start:start + 8]) + ","
            for start in range(0, len(values), 8)]
    fixture_header = "\n".join([
        "#pragma once",
        "#include <cstddef>",
        "namespace bp_fixture {",
        f"inline constexpr const char* kCaseId = \"{caseid}\";",
        f"inline constexpr const char* kSubjectId = \"{subject}\";",
        f"inline constexpr int kTimeSeconds = {time_s};",
        f"inline constexpr float kReferenceSbp = {c_float(reference[0])};",
        f"inline constexpr float kReferenceDbp = {c_float(reference[1])};",
        f"inline constexpr float kExpectedFloatSbp = {c_float(prediction[0])};",
        f"inline constexpr float kExpectedFloatDbp = {c_float(prediction[1])};",
        f"inline constexpr float kRawPpg[{len(raw)}] = {{",
        *rows,
        "};",
        "}",
        "",
    ])
    (args.out / "ppg_fixture.hpp").write_text(fixture_header)
    preprocess_header = "\n".join([
        "#pragma once",
        "#include <cstddef>",
        "namespace bp_config {",
        f"inline constexpr std::size_t kSampleRateHz = {config['sample_rate_hz']};",
        f"inline constexpr std::size_t kWindowSamples = {config['window_samples']};",
        "inline constexpr std::size_t kStrideSamples = kSampleRateHz;",
        f"inline constexpr float kPpgMean = {c_float(config['mean'])};",
        f"inline constexpr float kPpgStd = {c_float(config['std'])};",
        f"inline constexpr float kOutputScaleMmhg = {c_float(config['output_scale_mmhg'])};",
        "}",
        "",
    ])
    (args.out / "ppg_preprocess.hpp").write_text(preprocess_header)
    metadata = {
        "source": "VitalDB SNUADC/PLETH training window",
        "caseid": caseid,
        "subjectid": subject,
        "time_s": time_s,
        "window_samples": len(raw),
        "sample_rate_hz": config["sample_rate_hz"],
        "raw_ppg_sha256": hashlib.sha256(raw.tobytes()).hexdigest(),
        "reference_sbp_dbp_mmhg": reference,
        "float_model_sbp_dbp_mmhg": prediction.tolist(),
        "selection": "training subject nearest SBP 120 / DBP 80",
    }
    (args.out / "ppg_fixture.json").write_text(json.dumps(metadata, indent=2))
    print(json.dumps(metadata, indent=2))


if __name__ == "__main__":
    main()
