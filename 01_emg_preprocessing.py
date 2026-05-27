"""
EMG Signal Preprocessing Pipeline
Based on: "Machine Learning based Neuromuscular Disease Detection
           and Classification Using EMG Signal" (WIECON-ECE 2024)

Steps:
  1 — Load raw .asc files (163,840 samples, 5 s @ 32,768 Hz)
  2 — Filter:
        • Notch  50 Hz  → remove power-line noise / ECG harmonics
        • Notch 100 Hz  → remove 2nd harmonic
        • Bandpass 30–450 Hz → remove EEG (<30 Hz), ECG (<40 Hz), high-freq noise
  3 — Segment each recording into 5 × 1-second windows

Usage:
    from importlib import import_module
    run_pipeline = import_module("01_emg_preprocessing").run_pipeline
    records = run_pipeline()        # uses DATA_DIR from 00_emg_config
"""

import os
import re
import glob
import importlib
import numpy as np
from collections import Counter
from scipy.signal import butter, filtfilt, iirnotch

_cfg = importlib.import_module("00_emg_config")
DATA_DIR    = _cfg.DATA_DIR
FS          = _cfg.FS
N_SAMPLES   = _cfg.N_SAMPLES
N_SEGMENTS  = _cfg.N_SEGMENTS
SEG_LEN     = _cfg.SEG_LEN
BP_LOW      = _cfg.BP_LOW
BP_HIGH     = _cfg.BP_HIGH
BP_ORDER    = _cfg.BP_ORDER
NOTCH_FREQS = _cfg.NOTCH_FREQS
NOTCH_Q     = _cfg.NOTCH_Q
CLASSES     = _cfg.CLASSES


# ─────────────────────────────────────────────────────────────────────────────
# STEP 1 — DATA LOADING
# ─────────────────────────────────────────────────────────────────────────────
def load_signal(filepath: str) -> np.ndarray:
    """
    Read one .asc file → 1-D float64 array.
    Handles adjacent negative numbers without separator,
    e.g. '-7741.4000-11768.8000' → '-7741.4000 -11768.8000'.
    """
    with open(filepath, "r") as fh:
        content = fh.read()
    content = re.sub(r"(\d)-", r"\1 -", content)
    return np.fromstring(content, sep=" ")


def load_all_records(data_dir: str = DATA_DIR) -> list[dict]:
    """
    Walk Healthy / Myopathy / Neuropathy sub-folders.

    Returns a list of dicts, each with:
        signal   : np.ndarray  — raw signal trimmed to exactly 5 s
        label    : int         — 0 Healthy | 1 Myopathy | 2 Neuropathy
        class    : str
        filename : str
        filepath : str

    Recordings shorter than 5 s are skipped automatically
    (1 known Neuropathy file in the Mendeley dataset).
    """
    records = []
    for class_name, label in CLASSES.items():
        folder = os.path.join(data_dir, class_name)
        files  = sorted(glob.glob(os.path.join(folder, "*.asc")))
        for fp in files:
            sig = load_signal(fp)
            if sig.size < N_SAMPLES:
                print(f"  [SKIP] {os.path.basename(fp)} — {sig.size} samples < {N_SAMPLES}")
                continue
            records.append({
                "signal":   sig[:N_SAMPLES],
                "label":    label,
                "class":    class_name,
                "filename": os.path.basename(fp),
                "filepath": fp,
            })
    return records


# ─────────────────────────────────────────────────────────────────────────────
# STEP 2 — FILTERING
# ─────────────────────────────────────────────────────────────────────────────
def _make_bandpass(low: float, high: float, fs: int, order: int):
    nyq  = fs / 2.0
    b, a = butter(order, [low / nyq, high / nyq], btype="band")
    return b, a


def _make_notch(freq: float, fs: int, Q: float):
    b, a = iirnotch(freq / (fs / 2.0), Q)
    return b, a


def filter_signal(raw: np.ndarray, fs: int = FS) -> np.ndarray:
    """
    Two-stage zero-phase filter chain (uses filtfilt — no phase distortion).

    Stage A — Notch filters
        50 Hz  : power-line noise + ECG harmonics
        100 Hz : 2nd harmonic of power-line

    Stage B — Butterworth bandpass (order 3)
        High-pass 30 Hz : removes EEG (0.5–30 Hz) and ECG (0.5–40 Hz)
        Low-pass 450 Hz : removes high-frequency noise above the EMG band

    Returns clean EMG signal in the 30–450 Hz window.
    """
    sig = raw.copy()

    for f in NOTCH_FREQS:
        b, a = _make_notch(f, fs, Q=NOTCH_Q)
        sig  = filtfilt(b, a, sig)

    b, a = _make_bandpass(BP_LOW, BP_HIGH, fs, order=BP_ORDER)
    sig  = filtfilt(b, a, sig)

    return sig


# ─────────────────────────────────────────────────────────────────────────────
# STEP 3 — SEGMENTATION
# ─────────────────────────────────────────────────────────────────────────────
def segment_signal(sig: np.ndarray,
                   seg_len: int = SEG_LEN,
                   n_seg:   int = N_SEGMENTS) -> list[np.ndarray]:
    """Split signal into n_seg non-overlapping 1-second windows."""
    return [sig[i * seg_len:(i + 1) * seg_len] for i in range(n_seg)]


# ─────────────────────────────────────────────────────────────────────────────
# FULL PIPELINE
# ─────────────────────────────────────────────────────────────────────────────
def run_pipeline(data_dir: str = DATA_DIR) -> list[dict]:
    """
    Execute Steps 1–3 and return the processed records list.

    Each record dict is extended with:
        filtered     : np.ndarray          — filtered full signal
        segments     : list[np.ndarray]    — 5 × filtered 1-second segments
        segments_raw : list[np.ndarray]    — 5 × raw 1-second segments (for display)
    """
    print("=" * 60)
    print("STEP 1 — Loading raw signals …")
    records = load_all_records(data_dir)
    counts  = Counter(r["class"] for r in records)
    for cls, n in counts.items():
        print(f"  {cls:12s}: {n:3d} recordings")
    print(f"  {'TOTAL':12s}: {len(records):3d} recordings")

    print(f"\nSTEP 2 — Filtering  [Notch {NOTCH_FREQS} Hz  +  BP {BP_LOW}–{BP_HIGH} Hz] …")
    for rec in records:
        rec["filtered"] = filter_signal(rec["signal"])

    print(f"\nSTEP 3 — Segmenting into {N_SEGMENTS} × 1-second windows …")
    for rec in records:
        rec["segments"]     = segment_signal(rec["filtered"])
        rec["segments_raw"] = segment_signal(rec["signal"])

    total_seg = sum(len(r["segments"]) for r in records)
    print(f"  Total segments: {total_seg}")
    print("=" * 60)
    return records


# ─────────────────────────────────────────────────────────────────────────────
# STANDALONE RUN (no GUI)
# ─────────────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    records = run_pipeline()
    print("\nDone. Access records list for downstream tasks (feature extraction, etc.)")
