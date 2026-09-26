"""Recover sample-level PPG alignment and extract matching VitalDB lead-II ECG.

The existing NPZ records integer time_s only. Do not use that value directly
for PAT: the original PPG's fractional offset must be recovered by matching
the saved waveform to the public PLETH track.
"""

import argparse
import csv
import gzip
import io
import json
import time
import urllib.request
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.signal import correlate


TRACKS_URL = "https://api.vitaldb.net/trks"
TRACK_URL = "https://api.vitaldb.net/{}"


def get_bytes(url, tries=4):
    for attempt in range(tries):
        try:
            with urllib.request.urlopen(url, timeout=120) as response:
                return response.read()
        except Exception:
            if attempt == tries - 1:
                raise
            time.sleep(2 ** attempt)


def get_track_map(path):
    if not path.exists():
        data = get_bytes(TRACKS_URL)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(data)
    with gzip.open(path, "rt") as f:
        return {(r["caseid"], r["tname"]): r["tid"] for r in csv.DictReader(f)}


def load_wave(tid):
    compressed = get_bytes(TRACK_URL.format(tid))
    with gzip.GzipFile(fileobj=io.BytesIO(compressed)) as stream:
        # VitalDB wave CSV has sparse Time values, but one value per row.
        values = pd.read_csv(stream, usecols=[1], dtype=np.float32).iloc[:, 0].to_numpy()
    return values


def match_one(ppg, saved, second):
    # The source PPG and saved PPG both have 125-Hz-equivalent sampling.
    # Permit several seconds of nominal-timestamp drift.
    left = max(0, int((second - 5) * 125))
    right = min(len(ppg), int((second + 16) * 125))
    candidate = ppg[left:right]
    if len(candidate) < 1250:
        return None
    valid = np.isfinite(candidate)
    if valid.mean() < 0.98 or not np.isfinite(saved).all():
        return None
    candidate = np.interp(np.arange(len(candidate)), np.flatnonzero(valid), candidate[valid])
    # Covariance maximum locates the candidate; verify full normalized
    # correlation afterward to reject periodic false matches.
    c = correlate(candidate - candidate.mean(), saved - saved.mean(), mode="valid", method="fft")
    top = np.argpartition(c, -min(8, len(c)))[-min(8, len(c)):]
    matches = [(float(np.corrcoef(candidate[j:j + 1250], saved)[0, 1]), int(j)) for j in top]
    corr, j = max(matches)
    if not np.isfinite(corr) or corr < 0.96:
        return None
    return 4 * (left + j), corr


def process_case(caseid, rows, source, tracks, output):
    dest = output / f"{caseid}.npz"
    if dest.exists():
        return caseid, "exists", 0
    ppg_tid = tracks.get((caseid, "SNUADC/PLETH"))
    ecg_tid = tracks.get((caseid, "SNUADC/ECG_II"))
    if not ppg_tid or not ecg_tid:
        return caseid, "no_track", 0
    ppg = load_wave(ppg_tid)[::4]
    ecg = load_wave(ecg_tid)
    indices, windows, correlations = [], [], []
    for row in rows:
        found = match_one(ppg, source["ppg_signals"][row], int(source["time_s"][row]))
        if found is None:
            continue
        start, corr = found
        raw = ecg[start:start + 5000]
        if len(raw) != 5000 or np.isfinite(raw).mean() < 0.98:
            continue
        raw = raw[::4]
        if not np.isfinite(raw).all():
            valid = np.isfinite(raw)
            raw = np.interp(np.arange(len(raw)), np.flatnonzero(valid), raw[valid])
        if np.std(raw) < 0.015:
            continue
        indices.append(row)
        windows.append(raw.astype(np.float32))
        correlations.append(corr)
    if indices:
        np.savez_compressed(dest, row_index=np.asarray(indices, np.int32),
                            ecg_signals=np.stack(windows),
                            ppg_match_correlation=np.asarray(correlations, np.float32))
    return caseid, "ok", len(indices)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--train", type=Path, required=True)
    p.add_argument("--test", type=Path, required=True)
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--workers", type=int, default=4)
    p.add_argument("--max-cases", type=int, default=0)
    p.add_argument("--part", choices=("train", "test", "both"), default="both")
    args = p.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    tracks = get_track_map(args.out / "trks.csv.gz")
    for name, path in (("train", args.train), ("test", args.test)):
        if args.part != "both" and name != args.part:
            continue
        with np.load(path, allow_pickle=False) as z:
            source = {k: z[k] for k in ("ppg_signals", "caseids", "time_s")}
        by_case = {}
        for i, c in enumerate(source["caseids"].astype(str)):
            by_case.setdefault(c, []).append(i)
        selected = sorted(by_case, key=lambda c: int(c))
        if args.max_cases:
            selected = selected[:args.max_cases]
        output = args.out / name
        output.mkdir(exist_ok=True)
        print(f"{name}: {len(selected)} cases", flush=True)
        with ThreadPoolExecutor(max_workers=args.workers) as pool:
            future = {pool.submit(process_case, c, by_case[c], source, tracks, output): c
                      for c in selected}
            for n, task in enumerate(as_completed(future), 1):
                c = future[task]
                try:
                    caseid, status, count = task.result()
                    print(f"{name} {n}/{len(selected)} case={caseid} {status} windows={count}", flush=True)
                except Exception as exc:
                    print(f"{name} {n}/{len(selected)} case={c} error={exc}", flush=True)


if __name__ == "__main__":
    main()
