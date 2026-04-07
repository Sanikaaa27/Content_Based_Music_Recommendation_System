"""
build_song_vectors.py
----------------------
Builds the final song vectors used by the recommendation system.

For each song the vector contains:
  • MFCC features     → 91 values  (normalized)
  • Genre probability → 10 values  (softmax output of trained MLP)
  • Tempo             →  1 value   (normalized)
Total: 102 values per song.

Saves: features/song_vectors.csv
"""

import os
import pickle
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler

# ── Paths ──────────────────────────────────────────────────────────────────────
BASE_DIR     = os.path.join(os.path.dirname(__file__), '..')
FEATURES_CSV = os.path.join(BASE_DIR, 'features', 'mfcc_features.csv')
OUTPUT_CSV   = os.path.join(BASE_DIR, 'features', 'song_vectors.csv')

MODEL_PATH   = os.path.join(BASE_DIR, 'models', 'genre_model.pkl')
SCALER_PATH  = os.path.join(BASE_DIR, 'models', 'scaler.pkl')
ENCODER_PATH = os.path.join(BASE_DIR, 'models', 'encoder.pkl')


def load_models():
    with open(MODEL_PATH,   'rb') as f: model   = pickle.load(f)
    with open(SCALER_PATH,  'rb') as f: scaler  = pickle.load(f)
    with open(ENCODER_PATH, 'rb') as f: encoder = pickle.load(f)
    return model, scaler, encoder


def main():
    df = pd.read_csv(FEATURES_CSV)
    model, scaler, encoder = load_models()

    mfcc_cols = [c for c in df.columns if c.startswith('mfcc')]
    X_mfcc    = df[mfcc_cols].values           # (N, 91)
    tempo_raw = df['tempo'].values.reshape(-1, 1)

    # ── Scale MFCC (same scaler used in training) ───────────────────────────────
    X_mfcc_scaled = scaler.transform(X_mfcc)

    # ── Genre probability vector (10 values per song) ───────────────────────────
    genre_probs = model.predict_proba(X_mfcc_scaled)  # (N, 10)

    # ── Normalise MFCC and tempo to [0,1] for distance calculation ──────────────
    mfcc_scaler  = MinMaxScaler()
    tempo_scaler = MinMaxScaler()

    X_mfcc_norm  = mfcc_scaler.fit_transform(X_mfcc_scaled)
    tempo_norm   = tempo_scaler.fit_transform(tempo_raw)

    # ── Assemble final vector ────────────────────────────────────────────────────
    # Columns: mfcc_0 … mfcc_90 | genre_0 … genre_9 | tempo
    genre_names = encoder.classes_
    mfcc_norm_df  = pd.DataFrame(X_mfcc_norm,
                                  columns=[f'mfcc_norm_{i}' for i in range(91)])
    genre_prob_df = pd.DataFrame(genre_probs,
                                  columns=[f'genre_prob_{g}' for g in genre_names])
    tempo_norm_df = pd.DataFrame(tempo_norm, columns=['tempo_norm'])

    result = pd.concat([
        df[['file_name', 'genre']].reset_index(drop=True),
        mfcc_norm_df,
        genre_prob_df,
        tempo_norm_df
    ], axis=1)

    result.to_csv(OUTPUT_CSV, index=False)
    print(f"✅ Saved {len(result)} song vectors → {OUTPUT_CSV}")
    print(f"   Shape: {result.shape}")
    print(f"   Columns: 2 meta + 91 mfcc + 10 genre + 1 tempo = 104 total")


if __name__ == '__main__':
    main()