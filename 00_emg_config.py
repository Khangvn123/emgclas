"""
Shared configuration — imported by both emg_preprocessing.py and emg_gui.py
"""

DATA_DIR    = r"c:/Users/22200/Desktop/EMG_Attemp2_Fuzzy_SNN hybrid/EMG_Attemp2/02_Biceps Brachii"

# Signal parameters
FS          = 32_768    # Hz — sampling frequency
DURATION_S  = 5         # seconds per recording
N_SAMPLES   = FS * DURATION_S   # 163,840
N_SEGMENTS  = 5
SEG_LEN     = FS * 1    # 32,768 samples per 1-second segment

# Butterworth bandpass
BP_LOW      = 16        # Hz — high-pass: removes EEG (<30 Hz) & ECG (<40 Hz)
BP_HIGH     = 5000      # Hz — low-pass:  removes high-freq noise
BP_ORDER    = 3

# Notch filters
NOTCH_FREQS = [50, 100] # Hz — power-line fundamental + 2nd harmonic
NOTCH_Q     = 30        # quality factor

CLASSES = {
    "Healthy":    0,
    "Myopathy":   1,
    "Neuropathy": 2,
}

CLASS_COLORS = {
    "Healthy":    "#2ca02c",
    "Myopathy":   "#d62728",
    "Neuropathy": "#1f77b4",
}
