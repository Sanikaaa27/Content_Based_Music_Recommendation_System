
import os
import sys
import pickle
import tempfile
import numpy as np
import pandas as pd
import librosa
from flask import Flask, request, jsonify, send_from_directory
from flask_cors import CORS
from scipy.stats import skew, kurtosis
from sklearn.preprocessing import MinMaxScaler

# ── make src imports work ──────────────────────────────────────────────────────
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

app = Flask(__name__, static_folder='static')
CORS(app)

# ── Paths ──────────────────────────────────────────────────────────────────────
BASE_DIR     = os.path.dirname(__file__)
VECTORS_CSV  = os.path.join(BASE_DIR, 'features', 'song_vectors.csv')
MODEL_PATH   = os.path.join(BASE_DIR, 'models', 'genre_model.pkl')
SCALER_PATH  = os.path.join(BASE_DIR, 'models', 'scaler.pkl')
ENCODER_PATH = os.path.join(BASE_DIR, 'models', 'encoder.pkl')

# ── Constants ──────────────────────────────────────────────────────────────────
N_MFCC           = 13
SAMPLE_RATE      = 22050
HOP_SIZE         = 512
WINDOW_SIZE      = 2048
SEGMENT_DURATION = 3
SAMPLES_PER_SEG  = SAMPLE_RATE * SEGMENT_DURATION

STRATEGIES = {
    'PrioritizeMFCC':  (4, 1, 1),
    'PrioritizeGenre': (1, 4, 1),
    'PrioritizeTempo': (1, 1, 4),
    'Unbalanced':      (2, 3, 1),
    'Random':          (0, 0, 0),
}

GENRE_COLORS = {
    'blues':     '#1E90FF',
    'classical': '#9B59B6',
    'country':   '#E67E22',
    'disco':     '#E91E8C',
    'hiphop':    '#2ECC71',
    'jazz':      '#F39C12',
    'metal':     '#E74C3C',
    'pop':       '#FF69B4',
    'reggae':    '#27AE60',
    'rock':      '#E74C3C',
}

# ── Load models & database once at startup ─────────────────────────────────────
print("Loading models and database …")
with open(MODEL_PATH,   'rb') as f: model   = pickle.load(f)
with open(SCALER_PATH,  'rb') as f: scaler  = pickle.load(f)
with open(ENCODER_PATH, 'rb') as f: encoder = pickle.load(f)
db = pd.read_csv(VECTORS_CSV)

mfcc_cols  = [c for c in db.columns if c.startswith('mfcc_norm_')]
genre_cols = [c for c in db.columns if c.startswith('genre_prob_')]
db_mfcc    = db[mfcc_cols].values
db_genre   = db[genre_cols].values
db_tempo   = db[['tempo_norm']].values
genre_names = list(encoder.classes_)
print(f"✅ Loaded {len(db)} songs")


# ── Feature extraction ─────────────────────────────────────────────────────────

def extract_features(file_path):
    y, sr = librosa.load(file_path, sr=SAMPLE_RATE, mono=True)
    num_segments = max(1, len(y) // SAMPLES_PER_SEG)

    all_mfccs = []
    for i in range(num_segments):
        start   = i * SAMPLES_PER_SEG
        segment = y[start: start + SAMPLES_PER_SEG]
        if len(segment) < SAMPLES_PER_SEG:
            segment = np.pad(segment, (0, SAMPLES_PER_SEG - len(segment)))
        mfcc = librosa.feature.mfcc(y=segment, sr=sr, n_mfcc=N_MFCC,
                                     n_fft=WINDOW_SIZE, hop_length=HOP_SIZE)
        all_mfccs.append(mfcc)

    mfcc_concat = np.concatenate(all_mfccs, axis=1)
    feats = []
    for i in range(N_MFCC):
        c = mfcc_concat[i]
        feats.extend([np.mean(c), np.min(c), np.max(c), np.median(c),
                      np.std(c), float(skew(c)), float(kurtosis(c))])

    tempo, _ = librosa.beat.beat_track(y=y, sr=sr, hop_length=HOP_SIZE)
    tempo_val = float(np.atleast_1d(tempo)[0])

    return np.array(feats), tempo_val


def vectorise(mfcc_raw, tempo_raw):
    mfcc_scaled  = scaler.transform(mfcc_raw.reshape(1, -1))
    genre_probs  = model.predict_proba(mfcc_scaled)[0]

    mfcc_scaler  = MinMaxScaler()
    tempo_scaler = MinMaxScaler()
    mfcc_scaler.fit(db_mfcc)
    tempo_scaler.fit(db_tempo)

    mfcc_norm  = mfcc_scaler.transform(mfcc_scaled)[0]
    tempo_norm = float(tempo_scaler.transform([[tempo_raw]])[0][0])

    return mfcc_norm, genre_probs, tempo_norm


def feature_prob(query_vec, db_mat):
    diff = np.abs(db_mat - query_vec)
    return np.clip(1.0 - diff.mean(axis=1), 0, 1)


def recommend(mfcc_norm, genre_probs, tempo_norm, strategy, top_n=5):
    p_mfcc  = feature_prob(mfcc_norm,             db_mfcc)
    p_genre = feature_prob(genre_probs,            db_genre)
    p_tempo = feature_prob(np.array([tempo_norm]), db_tempo)

    if strategy == 'Random':
        p_strat = np.random.random(len(db))
    else:
        wm, wg, wt = STRATEGIES[strategy]
        p_strat = (p_mfcc * wm + p_genre * wg + p_tempo * wt) / (wm + wg + wt)

    results = db[['file_name', 'genre']].copy()
    results['p_mfcc']     = p_mfcc
    results['p_genre']    = p_genre
    results['p_tempo']    = p_tempo
    results['p_strategy'] = p_strat

    results = results.sort_values('p_strategy', ascending=False).head(top_n)
    return results.to_dict('records')


# ── Routes ─────────────────────────────────────────────────────────────────────

@app.route('/')
def index():
    return send_from_directory('static', 'index.html')


@app.route('/api/recommend', methods=['POST'])
def api_recommend():
    if 'file' not in request.files:
        return jsonify({'error': 'No file uploaded'}), 400

    f        = request.files['file']
    strategy = request.form.get('strategy', 'PrioritizeMFCC')
    top_n    = int(request.form.get('top_n', 5))

    # Save to temp file
    suffix = os.path.splitext(f.filename)[1] or '.wav'
    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
        f.save(tmp.name)
        tmp_path = tmp.name

    try:
        mfcc_raw, tempo_raw           = extract_features(tmp_path)
        mfcc_norm, genre_probs, t_norm = vectorise(mfcc_raw, tempo_raw)

        # Predicted genre of query
        pred_idx   = int(np.argmax(genre_probs))
        pred_genre = genre_names[pred_idx]
        pred_conf  = float(genre_probs[pred_idx])

        recs = recommend(mfcc_norm, genre_probs, t_norm, strategy, top_n)

        # Enrich with color
        for r in recs:
            r['color'] = GENRE_COLORS.get(r['genre'], '#888')
            r['p_mfcc']     = round(r['p_mfcc'], 4)
            r['p_genre']    = round(r['p_genre'], 4)
            r['p_tempo']    = round(r['p_tempo'], 4)
            r['p_strategy'] = round(r['p_strategy'], 4)

        # Genre probability breakdown for query
        genre_breakdown = [
            {'genre': g, 'prob': round(float(p), 4),
             'color': GENRE_COLORS.get(g, '#888')}
            for g, p in zip(genre_names, genre_probs)
        ]
        genre_breakdown.sort(key=lambda x: x['prob'], reverse=True)

        return jsonify({
            'query_file':       f.filename,
            'predicted_genre':  pred_genre,
            'genre_confidence': round(pred_conf * 100, 1),
            'tempo':            round(tempo_raw, 1),
            'strategy':         strategy,
            'genre_breakdown':  genre_breakdown,
            'recommendations':  recs,
        })

    except Exception as e:
        return jsonify({'error': str(e)}), 500
    finally:
        os.unlink(tmp_path)


@app.route('/api/strategies')
def api_strategies():
    return jsonify(list(STRATEGIES.keys()))


@app.route('/api/stats')
def api_stats():
    genre_counts = db['genre'].value_counts().to_dict()
    return jsonify({'total_songs': len(db), 'genre_counts': genre_counts})


if __name__ == '__main__':
    os.makedirs('static', exist_ok=True)
    app.run(debug=True, port=5000)
