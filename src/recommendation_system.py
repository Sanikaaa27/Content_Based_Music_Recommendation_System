"""
recommendation_system.py
--------------------------
Content-based music recommendation system replicating the paper:
  "Attributes Relevance in Content-Based Music Recommendation System"
  Kostrzewa et al., Applied Sciences 2024.

Implements all 5 strategies (Section 3.2):
  1. PrioritizeMFCC    – MFCC ×4, Genre ×1, Tempo ×1
  2. PrioritizeGenre   – Genre ×4, MFCC ×1, Tempo ×1
  3. PrioritizeTempo   – Tempo ×4, MFCC ×1, Genre ×1
  4. Unbalanced        – Genre ×3, MFCC ×2, Tempo ×1
  5. Random            – control baseline

Probability formula (paper Section 3.2):
  P_feature = 1 − (sum |diff| / vector_length)
  P_strategy = weighted average of P_MFCC, P_Genre, P_Tempo
"""

import os
import pickle
import random
import numpy as np
import pandas as pd
import librosa
from scipy.stats       import skew, kurtosis
from sklearn.preprocessing import MinMaxScaler

# ── Paths ──────────────────────────────────────────────────────────────────────
BASE_DIR     = os.path.join(os.path.dirname(__file__), '..')
VECTORS_CSV  = os.path.join(BASE_DIR, 'features', 'song_vectors.csv')
MODEL_PATH   = os.path.join(BASE_DIR, 'models', 'genre_model.pkl')
SCALER_PATH  = os.path.join(BASE_DIR, 'models', 'scaler.pkl')
ENCODER_PATH = os.path.join(BASE_DIR, 'models', 'encoder.pkl')

# ── Strategy weight definitions ────────────────────────────────────────────────
# (weight_mfcc, weight_genre, weight_tempo)
STRATEGIES = {
    'PrioritizeMFCC':  (4, 1, 1),
    'PrioritizeGenre': (1, 4, 1),
    'PrioritizeTempo': (1, 1, 4),
    'Unbalanced':      (2, 3, 1),
    'Random':          (0, 0, 0),   # handled separately
}

# ── Audio extraction parameters (same as extract_features.py) ─────────────────
N_MFCC           = 13
SAMPLE_RATE      = 22050
HOP_SIZE         = 512
WINDOW_SIZE      = 2048
SEGMENT_DURATION = 3
SAMPLES_PER_SEGMENT = SAMPLE_RATE * SEGMENT_DURATION


# ══════════════════════════════════════════════════════════════════════════════
# Feature extraction (for a NEW query song not in the database)
# ══════════════════════════════════════════════════════════════════════════════

def extract_mfcc_features(file_path: str) -> np.ndarray:
    """Returns a 91-dim MFCC feature vector (same logic as extract_features.py)."""
    y, sr = librosa.load(file_path, sr=SAMPLE_RATE, mono=True)
    num_segments = max(1, len(y) // SAMPLES_PER_SEGMENT)

    all_mfccs = []
    for i in range(num_segments):
        start   = i * SAMPLES_PER_SEGMENT
        segment = y[start: start + SAMPLES_PER_SEGMENT]
        if len(segment) < SAMPLES_PER_SEGMENT:
            segment = np.pad(segment, (0, SAMPLES_PER_SEGMENT - len(segment)))
        mfcc = librosa.feature.mfcc(y=segment, sr=sr, n_mfcc=N_MFCC,
                                     n_fft=WINDOW_SIZE, hop_length=HOP_SIZE)
        all_mfccs.append(mfcc)

    mfcc_concat = np.concatenate(all_mfccs, axis=1)

    feats = []
    for i in range(N_MFCC):
        c = mfcc_concat[i]
        feats.extend([
            np.mean(c), np.min(c), np.max(c), np.median(c),
            np.std(c), float(skew(c)), float(kurtosis(c))
        ])
    return np.array(feats)   # (91,)


def extract_tempo(file_path: str) -> float:
    y, sr = librosa.load(file_path, sr=SAMPLE_RATE, mono=True)
    tempo, _ = librosa.beat.beat_track(y=y, sr=sr, hop_length=HOP_SIZE)
    return float(np.atleast_1d(tempo)[0])


# ══════════════════════════════════════════════════════════════════════════════
# Recommendation System Class
# ══════════════════════════════════════════════════════════════════════════════

class MusicRecommender:
    """
    Content-based music recommendation system.

    Usage:
        rec = MusicRecommender()
        rec.load_database()                        # load pre-built vectors
        results = rec.recommend('path/to/song.wav') # get recommendations
    """

    def __init__(self):
        self.db: pd.DataFrame | None = None
        self._load_models()

    def _load_models(self):
        with open(MODEL_PATH,   'rb') as f: self.model   = pickle.load(f)
        with open(SCALER_PATH,  'rb') as f: self.scaler  = pickle.load(f)
        with open(ENCODER_PATH, 'rb') as f: self.encoder = pickle.load(f)
        self.genre_names = list(self.encoder.classes_)

    # ── Database ──────────────────────────────────────────────────────────────

    def load_database(self, vectors_csv: str = VECTORS_CSV):
        """Load pre-computed song vectors from CSV."""
        self.db = pd.read_csv(vectors_csv)
        print(f"Loaded database: {len(self.db)} songs")

    def _get_db_vectors(self):
        """Split database into MFCC, Genre, Tempo numpy arrays."""
        mfcc_cols  = [c for c in self.db.columns if c.startswith('mfcc_norm_')]
        genre_cols = [c for c in self.db.columns if c.startswith('genre_prob_')]
        mfcc_mat   = self.db[mfcc_cols].values   # (N, 91)
        genre_mat  = self.db[genre_cols].values  # (N, 10)
        tempo_vec  = self.db[['tempo_norm']].values  # (N, 1)
        return mfcc_mat, genre_mat, tempo_vec

    # ── Query song processing ─────────────────────────────────────────────────

    def _vectorise_query(self, file_path: str):
        """
        Extract and normalise features for a query song.
        Returns (mfcc_norm, genre_probs, tempo_norm) each as 1D arrays.
        """
        # MFCC raw
        mfcc_raw = extract_mfcc_features(file_path)          # (91,)
        tempo_raw = extract_tempo(file_path)                  # scalar

        # Scale MFCC (using training scaler)
        mfcc_scaled = self.scaler.transform(mfcc_raw.reshape(1, -1))  # (1,91)

        # Genre probabilities
        genre_probs = self.model.predict_proba(mfcc_scaled)[0]        # (10,)

        # Normalise MFCC and tempo to [0,1] using database stats
        db_mfcc_cols  = [c for c in self.db.columns if c.startswith('mfcc_norm_')]
        db_tempo_col  = 'tempo_norm'

        # For a fair comparison, re-fit MinMaxScaler on database + query point
        db_mfcc_mat  = self.db[db_mfcc_cols].values
        db_tempo_vec = self.db[db_tempo_col].values

        mfcc_scaler  = MinMaxScaler()
        tempo_scaler = MinMaxScaler()
        mfcc_scaler.fit(db_mfcc_mat)
        tempo_scaler.fit(db_tempo_vec.reshape(-1, 1))

        mfcc_norm  = mfcc_scaler.transform(mfcc_scaled)[0]         # (91,)
        tempo_norm = tempo_scaler.transform([[tempo_raw]])[0][0]   # scalar

        return mfcc_norm, genre_probs, tempo_norm

    # ── Similarity calculation ─────────────────────────────────────────────────

    @staticmethod
    def _feature_probability(query_vec: np.ndarray,
                              db_mat:    np.ndarray) -> np.ndarray:
        """
        P_feature = 1 - (sum|diff| / vector_length)  for each database song.
        Paper Section 3.2.
        """
        diff = np.abs(db_mat - query_vec)          # broadcast: (N, D)
        p    = 1.0 - diff.mean(axis=1)             # (N,)
        return np.clip(p, 0, 1)

    def _strategy_probability(self,
                               p_mfcc:  np.ndarray,
                               p_genre: np.ndarray,
                               p_tempo: np.ndarray,
                               strategy: str) -> np.ndarray:
        """Compute weighted P_strategy for all songs in the database."""
        if strategy == 'Random':
            return np.random.random(len(self.db))

        w_mfcc, w_genre, w_tempo = STRATEGIES[strategy]
        total = w_mfcc + w_genre + w_tempo
        return (p_mfcc * w_mfcc + p_genre * w_genre + p_tempo * w_tempo) / total

    # ── Public API ─────────────────────────────────────────────────────────────

    def recommend(self,
                  file_path: str,
                  strategy:  str = 'PrioritizeMFCC',
                  top_n:     int = 5,
                  exclude_self: bool = True) -> pd.DataFrame:
        """
        Recommend the top_n most similar songs for a given audio file.

        Parameters
        ----------
        file_path    : path to query .wav / .au file
        strategy     : one of STRATEGIES keys
        top_n        : number of recommendations to return
        exclude_self : if True, exclude the query file itself from results

        Returns
        -------
        DataFrame with columns: rank, file_name, genre, P_MFCC, P_Genre,
                                 P_Tempo, P_Strategy
        """
        assert self.db is not None, "Call load_database() first."
        assert strategy in STRATEGIES, f"Unknown strategy: {strategy}"

        print(f"\n🎵 Query : {os.path.basename(file_path)}")
        print(f"   Strategy: {strategy}")

        # Feature extraction
        mfcc_norm, genre_probs, tempo_norm = self._vectorise_query(file_path)

        # Database vectors
        db_mfcc, db_genre, db_tempo = self._get_db_vectors()

        # Per-feature probabilities
        p_mfcc  = self._feature_probability(mfcc_norm,            db_mfcc)
        p_genre = self._feature_probability(genre_probs,          db_genre)
        p_tempo = self._feature_probability(np.array([tempo_norm]), db_tempo)

        # Strategy score
        p_strat = self._strategy_probability(p_mfcc, p_genre, p_tempo, strategy)

        # Build results DataFrame
        results = self.db[['file_name', 'genre']].copy()
        results['P_MFCC']     = p_mfcc
        results['P_Genre']    = p_genre
        results['P_Tempo']    = p_tempo
        results['P_Strategy'] = p_strat

        # Optionally exclude the query song itself
        if exclude_self:
            query_name = os.path.basename(file_path)
            results    = results[results['file_name'] != query_name]

        results = results.sort_values('P_Strategy', ascending=False)
        results = results.head(top_n).reset_index(drop=True)
        results.insert(0, 'rank', results.index + 1)

        return results

    def recommend_all_strategies(self,
                                  file_path: str,
                                  top_n: int = 5) -> dict:
        """
        Run all 5 strategies and return a dict: strategy → DataFrame.
        Useful for reproducing the paper's comparative evaluation.
        """
        return {
            name: self.recommend(file_path, strategy=name, top_n=top_n)
            for name in STRATEGIES
        }


# ══════════════════════════════════════════════════════════════════════════════
# CLI demo
# ══════════════════════════════════════════════════════════════════════════════

if __name__ == '__main__':
    import sys

    if len(sys.argv) < 2:
        print("Usage: python recommendation_system.py <path_to_audio_file> [strategy] [top_n]")
        print(f"Available strategies: {list(STRATEGIES.keys())}")
        sys.exit(1)

    query_file = sys.argv[1]
    strategy   = sys.argv[2] if len(sys.argv) > 2 else 'PrioritizeMFCC'
    top_n      = int(sys.argv[3]) if len(sys.argv) > 3 else 5

    rec = MusicRecommender()
    rec.load_database()

    df_results = rec.recommend(query_file, strategy=strategy, top_n=top_n)

    print(f"\n{'='*60}")
    print(f"Top {top_n} recommendations ({strategy}):")
    print('='*60)
    print(df_results.to_string(index=False))

    # ── Compare all strategies ────────────────────────────────────────────────
    print(f"\n{'='*60}")
    print("All-strategy comparison (top 1 per strategy):")
    print('='*60)
    all_results = rec.recommend_all_strategies(query_file, top_n=1)
    for strat, df in all_results.items():
        row = df.iloc[0]
        print(f"  {strat:<20} → {row['file_name']}  [{row['genre']}]"
              f"  P={row['P_Strategy']:.4f}")