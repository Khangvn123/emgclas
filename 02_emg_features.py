"""
EMG Feature Extraction
Trich xuat 17 dac trung tu moi segment da loc:

  [Time domain]
    MAV  - Mean Absolute Value           : trung binh tri tuyet doi
    RMS  - Root Mean Square              : can bac hai trung binh binh phuong
    VAR  - Variance                      : phuong sai bien do (EMG variant)
    WL   - Waveform Length               : tong bien thien tuyet doi
    SD   - Standard Deviation            : do lech chuan
    ZC   - Zero Crossing                 : so lan tin hieu cat qua 0
    SSC  - Slope Sign Change             : so lan dao ham doi dau
    WAMP - Willison Amplitude            : so lan |diff| vuot nguong
    IEMG - Integrated EMG                : tong tri tuyet doi
    SSI  - Simple Square Integral        : tong binh phuong
    MV   - Mean Value                    : gia tri trung binh
    LOG  - Log Detector                  : exp(mean(log(|x|)))
    MFL  - Maximum Fractal Length        : log10(sqrt(sum(diff^2)))
    DAMV - Difference Absolute Mean Value: mean(|diff|)

  [Frequency domain]
    MNF  - Mean Frequency                : tan so trung binh cua PSD
    MDF  - Median Frequency              : tan so trung vi cua PSD
    PKF  - Peak Frequency                : tan so co cong suat lon nhat

Dau vao : records tu 01_emg_preprocessing.run_pipeline()
Dau ra  : features_17.csv  (n_samples x 17 + meta)
          layer1_voting_stats.json  (STATS cho 06_layer1_voting_snn.py)

Usage:
    python 02_emg_features.py
"""

import importlib
import json
import os

import numpy as np
import pandas as pd

_pre = importlib.import_module("01_emg_preprocessing")
run_pipeline = _pre.run_pipeline

_cfg = importlib.import_module("00_emg_config")
CLASSES = _cfg.CLASSES
FS      = _cfg.FS

# ─────────────────────────────────────────────────────────────────────────────
# FEATURE CONSTANTS
# ─────────────────────────────────────────────────────────────────────────────

# 7 dac trung co separation ratio cao nhat — dung boi 06_layer1_voting_snn.py
LAYER1_FEATURES = ['MAV', 'RMS', 'WL', 'SD', 'LOG', 'DAMV', 'IEMG']

# Ten file dau ra chinh — 06_layer1_voting_snn.py doc file nay
_HERE       = os.path.dirname(os.path.abspath(__file__))
OUTPUT_CSV  = os.path.join(_HERE, "features_17.csv")
OUTPUT_STATS = os.path.join(_HERE, "layer1_voting_stats.json")
OUTPUT_DIR  = _HERE


# ─────────────────────────────────────────────────────────────────────────────
# TIME-DOMAIN FEATURES
# ─────────────────────────────────────────────────────────────────────────────

def mav(seg: np.ndarray) -> float:
    """Mean Absolute Value — MAV = (1/N) * sum(|x_i|)"""
    return float(np.mean(np.abs(seg)))


def rms(seg: np.ndarray) -> float:
    """Root Mean Square — RMS = sqrt((1/N) * sum(x_i^2))"""
    return float(np.sqrt(np.mean(seg ** 2)))


def var(seg: np.ndarray) -> float:
    """Variance (EMG) — VAR = (1/N) * sum(x_i^2)  (khong tru mean)"""
    return float(np.mean(seg ** 2))


def wl(seg: np.ndarray) -> float:
    """Waveform Length — WL = sum(|x[i+1] - x[i]|)"""
    return float(np.sum(np.abs(np.diff(seg))))


def sd(seg: np.ndarray) -> float:
    """Standard Deviation — SD = std(x)"""
    return float(np.std(seg))


def zc(seg: np.ndarray, threshold: float = 0.0) -> float:
    """
    Zero Crossing — so lan tin hieu cat qua muc 0.
    ZC = sum( sign(x[i]) != sign(x[i+1])  AND  |x[i] - x[i+1]| > threshold )
    """
    s = seg.astype(float)
    sign_change = (s[:-1] * s[1:] < 0)
    above_thresh = np.abs(s[:-1] - s[1:]) > threshold
    return float(np.sum(sign_change & above_thresh))


def ssc(seg: np.ndarray, threshold: float = 0.0) -> float:
    """
    Slope Sign Change — so lan dao ham doi dau.
    SSC = sum( d[i]*d[i-1] < 0  AND  |d| > threshold )
    """
    d = np.diff(seg.astype(float))
    changes = np.sum(
        (d[:-1] * d[1:] < -threshold) |
        ((np.abs(d[:-1]) > threshold) & (np.abs(d[1:]) > threshold) & (d[:-1] * d[1:] < 0))
    )
    return float(changes)


def wamp(seg: np.ndarray, threshold: float = 10.0) -> float:
    """
    Willison Amplitude — so lan |x[i+1] - x[i]| vuot nguong.
    Threshold mac dinh 10 (don vi cung voi tin hieu).
    """
    return float(np.sum(np.abs(np.diff(seg.astype(float))) > threshold))


def iemg(seg: np.ndarray) -> float:
    """Integrated EMG — IEMG = sum(|x_i|)"""
    return float(np.sum(np.abs(seg)))


def ssi(seg: np.ndarray) -> float:
    """Simple Square Integral — SSI = sum(x_i^2)"""
    return float(np.sum(seg ** 2))


def mv(seg: np.ndarray) -> float:
    """Mean Value — MV = mean(x)"""
    return float(np.mean(seg))


def log_detector(seg: np.ndarray) -> float:
    """
    Log Detector — LOG = exp( (1/N) * sum(log(|x_i|)) )
    Do lech trung binh log-amplitude.
    """
    return float(np.exp(np.mean(np.log(np.abs(seg) + 1e-9))))


def mfl(seg: np.ndarray) -> float:
    """
    Maximum Fractal Length — MFL = log10( sqrt( sum((x[i+1]-x[i])^2) ) )
    Do do phuc tap fractal cua tin hieu.
    """
    return float(np.log10(np.sqrt(np.sum(np.diff(seg.astype(float)) ** 2)) + 1e-9))


def damv(seg: np.ndarray) -> float:
    """Difference Absolute Mean Value — DAMV = mean(|x[i+1] - x[i]|)"""
    return float(np.mean(np.abs(np.diff(seg.astype(float)))))


# ─────────────────────────────────────────────────────────────────────────────
# FREQUENCY-DOMAIN FEATURES
# ─────────────────────────────────────────────────────────────────────────────

def mnf(seg: np.ndarray) -> float:
    """Mean Frequency — MNF = sum(f_i * PSD_i) / sum(PSD_i)"""
    freqs = np.fft.rfftfreq(len(seg), 1.0 / FS)
    psd   = np.abs(np.fft.rfft(seg)) ** 2
    return float(np.sum(freqs * psd) / (np.sum(psd) + 1e-9))


def mdf(seg: np.ndarray) -> float:
    """Median Frequency — tan so tai do PSD tich luy dat 50%."""
    freqs  = np.fft.rfftfreq(len(seg), 1.0 / FS)
    psd    = np.abs(np.fft.rfft(seg)) ** 2
    cumsum = np.cumsum(psd)
    idx    = np.searchsorted(cumsum, cumsum[-1] / 2.0)
    idx    = min(idx, len(freqs) - 1)
    return float(freqs[idx])


def pkf(seg: np.ndarray) -> float:
    """Peak Frequency — PKF = tan so co cong suat PSD lon nhat."""
    freqs = np.fft.rfftfreq(len(seg), 1.0 / FS)
    psd   = np.abs(np.fft.rfft(seg)) ** 2
    return float(freqs[np.argmax(psd)])


# ─────────────────────────────────────────────────────────────────────────────
# FEATURE REGISTRY  (thu tu: MAV RMS VAR WL SD ZC SSC WAMP IEMG SSI MV LOG MFL DAMV MNF MDF PKF)
# ─────────────────────────────────────────────────────────────────────────────
FEATURE_FUNCS = {
    "MAV":  mav,
    "RMS":  rms,
    "VAR":  var,
    "WL":   wl,
    "SD":   sd,
    "ZC":   zc,
    "SSC":  ssc,
    "WAMP": wamp,
    "IEMG": iemg,
    "SSI":  ssi,
    "MV":   mv,
    "LOG":  log_detector,
    "MFL":  mfl,
    "DAMV": damv,
    "MNF":  mnf,
    "MDF":  mdf,
    "PKF":  pkf,
}

FEATURE_NAMES = list(FEATURE_FUNCS.keys())
# 17 features total


# ─────────────────────────────────────────────────────────────────────────────
# EXTRACTION
# ─────────────────────────────────────────────────────────────────────────────
def extract_segment(seg: np.ndarray) -> np.ndarray:
    """Trich xuat vector 17 dac trung tu 1 segment."""
    return np.array([fn(seg) for fn in FEATURE_FUNCS.values()])


def extract_all(records: list[dict]) -> tuple[np.ndarray, np.ndarray, list[dict]]:
    """
    Duyet qua tat ca records va segments, trich xuat dac trung.

    Tra ve:
        X    : np.ndarray  shape (n_samples, 17)
        y    : np.ndarray  shape (n_samples,)
        meta : list[dict]
    """
    X_rows, y_rows, meta = [], [], []

    for rec in records:
        for seg_idx, seg in enumerate(rec["segments"]):
            X_rows.append(extract_segment(seg))
            y_rows.append(rec["label"])
            meta.append({
                "class":    rec["class"],
                "label":    rec["label"],
                "filename": rec["filename"],
                "seg_idx":  seg_idx,
            })

    X = np.vstack(X_rows)
    y = np.array(y_rows)
    return X, y, meta


def to_dataframe(X: np.ndarray, y: np.ndarray, meta: list[dict]) -> pd.DataFrame:
    """Dong goi X, y va meta thanh DataFrame."""
    df = pd.DataFrame(X, columns=FEATURE_NAMES)
    df.insert(0, "class",    [m["class"]    for m in meta])
    df.insert(1, "label",    y)
    df.insert(2, "filename", [m["filename"] for m in meta])
    df.insert(3, "seg_idx",  [m["seg_idx"]  for m in meta])
    return df


# ─────────────────────────────────────────────────────────────────────────────
# STATS COMPUTATION  (phuc vu 06_layer1_voting_snn.py)
# ─────────────────────────────────────────────────────────────────────────────
def compute_class_stats(df: pd.DataFrame,
                        features: list[str] = LAYER1_FEATURES) -> dict:
    """
    Tinh mean va std cho tung class tren cac feature duoc chon.
    Healthy va Sick (Myopathy + Neuropathy gop lai) — khop voi STATS
    trong 06_layer1_voting_snn.py.

    Returns:
        {
          'Healthy': {'MAV': (mean, std), ...},
          'Sick':    {'MAV': (mean, std), ...},
        }
    """
    df_H = df[df['class'] == 'Healthy']
    df_S = df[df['class'].isin(['Myopathy', 'Neuropathy'])]

    stats: dict = {}
    for label, sub in [('Healthy', df_H), ('Sick', df_S)]:
        stats[label] = {}
        for feat in features:
            vals = sub[feat].values
            stats[label][feat] = (round(float(np.mean(vals)), 3),
                                  round(float(np.std(vals)),  3))
    return stats


def print_stats_for_layer1(statxs: dict) -> None:
    """In STATS dict theo dinh dang copy-paste vao 06_layer1_voting_snn.py."""
    print("\n  # --- Dan vao STATS trong 06_layer1_voting_snn.py ---")
    print("  STATS = {")
    for cls, feats in stats.items():
        print(f"      '{cls}': {{")
        for feat, (mean, std) in feats.items():
            print(f"          '{feat}':  ({mean},  {std}),")
        print("      },")
    print("  }")
    print("  # --- Het STATS ---\n")


# ─────────────────────────────────────────────────────────────────────────────
# STANDALONE RUN
# ─────────────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    print("Running preprocessing pipeline ...")
    records = run_pipeline()

    print(f"\nExtracting features {FEATURE_NAMES} ...")
    X, y, meta = extract_all(records)

    labels_inv = {v: k for k, v in CLASSES.items()}

    print(f"\n{'='*60}")
    print(f"  Total samples : {X.shape[0]}")
    print(f"  Features      : {X.shape[1]}  {FEATURE_NAMES}")
    print(f"  Label dist.   :")
    unique, counts = np.unique(y, return_counts=True)
    for lbl, cnt in zip(unique, counts):
        print(f"    {labels_inv[lbl]:12s} (label={lbl}): {cnt} samples")
    print(f"{'='*60}")

    df = to_dataframe(X, y, meta)

    # ── Luu features_17.csv (dau vao chinh cua 06_layer1_voting_snn.py) ───────
    df.to_csv(OUTPUT_CSV, index=False)
    print(f"\nSaved (all classes) : {OUTPUT_CSV}")

    # ── Luu tung lop ──────────────────────────────────────────────────────────
    class_file_map = {
        "Healthy":    "features_healthy.csv",
        "Myopathy":   "features_myopathy.csv",
        "Neuropathy": "features_neuropathy.csv",
    }
    for class_name, fname in class_file_map.items():
        df_cls = df[df["class"] == class_name].reset_index(drop=True)
        out_path = os.path.join(OUTPUT_DIR, fname)
        df_cls.to_csv(out_path, index=False)
        print(f"Saved ({class_name:10s}): {out_path}  [{len(df_cls)} rows]")

    # ── Tinh va luu STATS cho Layer 1 SNN ────────────────────────────────────
    print(f"\n  Tinh STATS cho Layer 1 ({LAYER1_FEATURES}) ...")
    stats = compute_class_stats(df, LAYER1_FEATURES)

    with open(OUTPUT_STATS, 'w', encoding='utf-8') as f:
        json.dump(stats, f, indent=4, ensure_ascii=False)
    print(f"  STATS da luu  : {OUTPUT_STATS}")

    print_stats_for_layer1(stats)

    # ── Thong ke mo ta ────────────────────────────────────────────────────────
    print(f"\n{'='*60}")
    print("Descriptive statistics per class (mean +/- std):")
    print(f"{'='*60}")
    for class_name in ["Healthy", "Myopathy", "Neuropathy"]:
        sub = df[df["class"] == class_name][FEATURE_NAMES]
        print(f"\n[{class_name}]")
        stat = sub.agg(["mean", "std"]).T
        stat.columns = ["mean", "std"]
        print(stat.round(4).to_string())
