"""Collect paired results and best checkpoints from completed seed runs."""

import argparse
import json
import shutil
from pathlib import Path

import numpy as np


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--runs", nargs="+", required=True,
                   help="seed:path entries, e.g. 42:D:/BP_Training/ecg_ppg/run_all_cases")
    p.add_argument("--out", type=Path, required=True)
    args = p.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    summaries = {}
    for item in args.runs:
        seed, path_text = item.split(":", 1)
        path = Path(path_text)
        training = json.loads((path / "metrics.json").read_text())
        paired = json.loads((path / "paired_metrics.json").read_text())
        if str(training["meta"]["seed"]) != seed:
            raise ValueError(f"Seed mismatch for {path}")
        for mode in training["models"]:
            dest = args.out / "weights" / f"seed{seed}" / f"{mode}.pt"
            dest.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(path / f"{mode}.pt", dest)
        summaries[seed] = {"training": training, "paired_test": paired}
    ordered = [summaries[key]["paired_test"] for key in sorted(summaries, key=int)]
    sizes = {(d["windows"], d["cases"]) for d in ordered}
    if len(sizes) != 1:
        raise ValueError("Runs do not share test counts")
    aggregates = {}
    for mode in ("ecg_residual", "ecg_delay_residual"):
        values = np.asarray([[
            d["modes"][mode][metric] - d["modes"]["ppg_only"][metric]
            for metric in ("sbp_mae", "dbp_mae")
        ] for d in ordered])
        correlation = np.asarray([[
            d["modes"][mode]["within_case_60_300s"][metric] -
            d["modes"]["ppg_only"]["within_case_60_300s"][metric]
            for metric in ("sbp_change_correlation", "dbp_change_correlation")
        ] for d in ordered])
        aggregates[mode] = {"mae_difference_by_seed_sbp_dbp": values.tolist(),
                            "mean_mae_difference_sbp_dbp": values.mean(axis=0).tolist(),
                            "change_correlation_difference_by_seed_sbp_dbp": correlation.tolist(),
                            "mean_change_correlation_difference_sbp_dbp":
                            correlation.mean(axis=0).tolist()}
    result = {"seeds": summaries, "aggregate": aggregates,
              "interpretation": "Exploratory repeated runs on the same inspected held-out subjects; seed variation is not independent external validation."}
    (args.out / "results.json").write_text(json.dumps(result, indent=2))
    print(json.dumps(aggregates, indent=2))


if __name__ == "__main__":
    main()
