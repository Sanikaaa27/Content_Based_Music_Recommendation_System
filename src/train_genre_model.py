"""
train_genre_model.py
---------------------
Trains a Multi-Layer Perceptron (MLP) for music genre classification.

Architecture (paper Figure 1):
    Input  → 91 neurons
    Dense  → 512 neurons, ReLU
    Dropout 0.3
    Dense  → 256 neurons, ReLU
    Dropout 0.3
    Dense  → 64  neurons, ReLU
    Dropout 0.3
    Output → 10  neurons  (one per genre)

Training: 200 epochs  |  Target accuracy: ~72%
Saves: models/genre_model.pkl, models/scaler.pkl, models/encoder.pkl
"""

import os
import pickle
import numpy as np
import pandas as pd
from sklearn.preprocessing      import LabelEncoder, StandardScaler
from sklearn.model_selection    import train_test_split
from sklearn.metrics            import accuracy_score, classification_report
from sklearn.neural_network     import MLPClassifier

# ── Paths ──────────────────────────────────────────────────────────────────────
BASE_DIR     = os.path.join(os.path.dirname(__file__), '..')
FEATURES_CSV = os.path.join(BASE_DIR, 'features', 'mfcc_features.csv')
MODELS_DIR   = os.path.join(BASE_DIR, 'models')

MODEL_PATH   = os.path.join(MODELS_DIR, 'genre_model.pkl')
SCALER_PATH  = os.path.join(MODELS_DIR, 'scaler.pkl')
ENCODER_PATH = os.path.join(MODELS_DIR, 'encoder.pkl')

# ── MFCC feature columns (91 values, no tempo, no metadata) ───────────────────
MFCC_COLS = [c for c in pd.read_csv(FEATURES_CSV, nrows=0).columns
             if c.startswith('mfcc')]   # 91 columns


def load_data():
    df = pd.read_csv(FEATURES_CSV)
    X = df[MFCC_COLS].values            # (N, 91)
    y_raw = df['genre'].values

    encoder = LabelEncoder()
    y = encoder.fit_transform(y_raw)    # integer labels 0-9

    print(f"Classes : {list(encoder.classes_)}")
    print(f"Dataset : {X.shape[0]} samples, {X.shape[1]} features\n")
    return X, y, encoder


def build_mlp() -> MLPClassifier:
    """
    Replicates the MLP from paper Figure 1.
    sklearn's MLPClassifier uses ReLU hidden activations by default
    and applies dropout-like regularisation via alpha (L2).
    For true dropout=0.3, we use a reasonably large alpha.
    """
    return MLPClassifier(
        hidden_layer_sizes=(512, 256, 64),
        activation='relu',
        solver='adam',
        alpha=1e-3,           # L2 regularisation (proxy for dropout)
        batch_size=64,
        learning_rate_init=1e-3,
        max_iter=200,         # 200 epochs as stated in paper
        early_stopping=True,
        validation_fraction=0.1,
        n_iter_no_change=20,
        random_state=42,
        verbose=True
    )


def main():
    os.makedirs(MODELS_DIR, exist_ok=True)

    # ── Load & split ────────────────────────────────────────────────────────────
    X, y, encoder = load_data()

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=42
    )

    # ── Scale features ──────────────────────────────────────────────────────────
    scaler  = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test  = scaler.transform(X_test)

    # ── Train ────────────────────────────────────────────────────────────────────
    print("Training MLP …\n")
    model = build_mlp()
    model.fit(X_train, y_train)

    # ── Evaluate ─────────────────────────────────────────────────────────────────
    y_pred = model.predict(X_test)
    acc    = accuracy_score(y_test, y_pred)
    print(f"\n✅ Test accuracy : {acc * 100:.2f}%  (paper target ≈ 72%)")
    print("\nClassification report:")
    print(classification_report(y_test, y_pred, target_names=encoder.classes_))

    # ── Save artefacts ───────────────────────────────────────────────────────────
    with open(MODEL_PATH,   'wb') as f: pickle.dump(model,   f)
    with open(SCALER_PATH,  'wb') as f: pickle.dump(scaler,  f)
    with open(ENCODER_PATH, 'wb') as f: pickle.dump(encoder, f)

    print(f"\nSaved → {MODEL_PATH}")
    print(f"Saved → {SCALER_PATH}")
    print(f"Saved → {ENCODER_PATH}")


if __name__ == '__main__':
    main()