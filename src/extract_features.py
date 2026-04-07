"""
extract_features.py
--------------------
Extracts MFCC (91 values) and Tempo (1 value) features from
all audio files in the GTZAN dataset and saves them to:
    features/mfcc_features.csv

Paper reference: Section 3.1 - Considered Features
- 13 MFCCs × 7 stats (mean, min, max, median, std, skewness, kurtosis) = 91 values
- Window size: 2048, Sample rate: 22050 Hz, Hop size: 512
- Computed on 3-second segments, then aggregated
"""

import os
import numpy as np
import pandas as pd
import librosa
from scipy.stats import skew, kurtosis

# ── Paths ──────────────────────────────────────────────────────────────────────
DATA_DIR     = os.path.join(os.path.dirname(__file__), '..', 'data', 'genres_original')
FEATURES_DIR = os.path.join(os.path.dirname(__file__), '..', 'features')
OUTPUT_CSV   = os.path.join(FEATURES_DIR, 'mfcc_features.csv')

# ── Parameters (from paper Section 3.1) ───────────────────────────────────────
N_MFCC      = 13
SAMPLE_RATE = 22050
HOP_SIZE    = 512
WINDOW_SIZE = 2048
SEGMENT_DURATION = 3   # seconds
SAMPLES_PER_SEGMENT = SAMPLE_RATE * SEGMENT_DURATION


def extract_features_from_file(file_path: str) -> dict | None:
    """
    Load an audio file, split into 3-second segments, compute MFCCs
    and tempo, then return aggregated statistics as a flat dict.
    Returns None if the file cannot be loaded.
    """
    try:
        y, sr = librosa.load(file_path, sr=SAMPLE_RATE, mono=True)
    except Exception as e:
        print(f"  [WARN] Could not load {file_path}: {e}")
        return None

    num_segments = len(y) // SAMPLES_PER_SEGMENT
    if num_segments == 0:
        print(f"  [WARN] File too short: {file_path}")
        return None

    # Collect per-segment MFCC matrices
    all_mfccs = []   # shape: (num_segments, N_MFCC, frames_per_segment)
    for seg_idx in range(num_segments):
        start = seg_idx * SAMPLES_PER_SEGMENT
        end   = start + SAMPLES_PER_SEGMENT
        segment = y[start:end]

        mfcc = librosa.feature.mfcc(
            y=segment,
            sr=sr,
            n_mfcc=N_MFCC,
            n_fft=WINDOW_SIZE,
            hop_length=HOP_SIZE
        )
        all_mfccs.append(mfcc)   # (N_MFCC, frames)

    # Stack all segments: (num_segments × N_MFCC × frames)
    # Flatten time axis: treat all frames across segments as one pool
    mfcc_concat = np.concatenate(all_mfccs, axis=1)  # (N_MFCC, total_frames)

    # Compute 7 statistics per MFCC coefficient  →  13 × 7 = 91 values
    feature_dict = {}
    for i in range(N_MFCC):
        coef = mfcc_concat[i]
        feature_dict[f'mfcc{i+1}_mean']   = np.mean(coef)
        feature_dict[f'mfcc{i+1}_min']    = np.min(coef)
        feature_dict[f'mfcc{i+1}_max']    = np.max(coef)
        feature_dict[f'mfcc{i+1}_median'] = np.median(coef)
        feature_dict[f'mfcc{i+1}_std']    = np.std(coef)
        feature_dict[f'mfcc{i+1}_skew']   = float(skew(coef))
        feature_dict[f'mfcc{i+1}_kurt']   = float(kurtosis(coef))

    # Tempo (beats per minute) – single value
    tempo, _ = librosa.beat.beat_track(y=y, sr=sr, hop_length=HOP_SIZE)
    feature_dict['tempo'] = float(np.atleast_1d(tempo)[0])

    return feature_dict


def main():
    os.makedirs(FEATURES_DIR, exist_ok=True)

    genres = sorted([
        g for g in os.listdir(DATA_DIR)
        if os.path.isdir(os.path.join(DATA_DIR, g))
    ])
    print(f"Found genres: {genres}\n")

    rows = []
    for genre in genres:
        genre_dir = os.path.join(DATA_DIR, genre)
        files = sorted([
            f for f in os.listdir(genre_dir)
            if f.endswith('.wav') or f.endswith('.au')
        ])
        print(f"Processing genre '{genre}' ({len(files)} files)...")
        for fname in files:
            fpath = os.path.join(genre_dir, fname)
            features = extract_features_from_file(fpath)
            if features is not None:
                features['file_name'] = fname
                features['genre']     = genre
                rows.append(features)

    df = pd.DataFrame(rows)
    # Put metadata columns first
    cols = ['file_name', 'genre'] + [c for c in df.columns if c not in ('file_name', 'genre')]
    df = df[cols]
    df.to_csv(OUTPUT_CSV, index=False)
    print(f"\n✅ Saved {len(df)} records → {OUTPUT_CSV}")
    print(f"   Shape: {df.shape}  (expected ~1000 rows × 93 columns)")


if __name__ == '__main__':
    main()