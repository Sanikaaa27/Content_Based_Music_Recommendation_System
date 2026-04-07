"""
plot_analysis.py  —  All 12 graphs for the Music Recommendation Project
------------------------------------------------------------------------
G1  Genre distribution bar chart
G2  Tempo box plot by genre
G3  Confusion matrix
G4  F1 / Precision / Recall per genre
G5  ROC-AUC curves
G6  Training loss curve
G7  Strategy average rating (horizontal bar)
G8  KDE score distribution
G9  Same-genre hit rate per strategy
G10 MFCC heatmap by genre
G11 t-SNE song vector clusters
G12 Genre probability radar chart

Run:  cd C:\\Users\\sanik\\Downloads\\RS_Project
      python plot_analysis.py
"""

import os, pickle, warnings
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
from matplotlib import rcParams
from sklearn.preprocessing   import label_binarize
from sklearn.metrics         import (confusion_matrix, classification_report,
                                     roc_curve, auc)
from sklearn.decomposition   import PCA
from sklearn.manifold        import TSNE
from sklearn.model_selection import train_test_split
from scipy.stats             import gaussian_kde

warnings.filterwarnings('ignore')

# ── Paths ──────────────────────────────────────────────────────────────────────
BASE_DIR     = r"C:\Users\sanik\Downloads\RS_Project"
FEATURES_CSV = os.path.join(BASE_DIR, 'features', 'mfcc_features.csv')
VECTORS_CSV  = os.path.join(BASE_DIR, 'features', 'song_vectors.csv')
MODEL_PATH   = os.path.join(BASE_DIR, 'models', 'genre_model.pkl')
SCALER_PATH  = os.path.join(BASE_DIR, 'models', 'scaler.pkl')
ENCODER_PATH = os.path.join(BASE_DIR, 'models', 'encoder.pkl')
OUT_DIR      = os.path.join(BASE_DIR, 'plots')
os.makedirs(OUT_DIR, exist_ok=True)

# ── Style ──────────────────────────────────────────────────────────────────────
rcParams['font.family']       = 'DejaVu Sans'
rcParams['axes.spines.top']   = False
rcParams['axes.spines.right'] = False
rcParams['axes.grid']         = True
rcParams['grid.alpha']        = 0.3
rcParams['grid.linewidth']    = 0.6

GENRES  = ['blues','classical','country','disco','hiphop',
           'jazz','metal','pop','reggae','rock']
PALETTE = ['#4472C4','#9B59B6','#E67E22','#E91E8C','#2ECC71',
           '#F39C12','#E74C3C','#FF69B4','#27AE60','#C0392B']
GENRE_COLORS = dict(zip(GENRES, PALETTE))

STRATEGIES = {
    'Random':          (0, 0, 0),
    'Unbalanced':      (2, 3, 1),
    'PrioritizeMFCC':  (4, 1, 1),
    'PrioritizeTempo': (1, 1, 4),
    'PrioritizeGenre': (1, 4, 1),
}
STRAT_COLORS = ['#FF4C4C','#FFC000','#4472C4','#A9D18E','#ED7D31']

# ── Load ───────────────────────────────────────────────────────────────────────
print("Loading data and models …")
with open(MODEL_PATH,   'rb') as f: model   = pickle.load(f)
with open(SCALER_PATH,  'rb') as f: scaler  = pickle.load(f)
with open(ENCODER_PATH, 'rb') as f: encoder = pickle.load(f)

feat_df   = pd.read_csv(FEATURES_CSV)
vec_df    = pd.read_csv(VECTORS_CSV)
mfcc_cols = [c for c in feat_df.columns if c.startswith('mfcc')]

X_raw = feat_df[mfcc_cols].values
y_raw = feat_df['genre'].values
y     = encoder.transform(y_raw)
X     = scaler.transform(X_raw)

X_tr, X_te, y_tr, y_te = train_test_split(
    X, y, test_size=0.2, stratify=y, random_state=42)
y_pred       = model.predict(X_te)
y_pred_proba = model.predict_proba(X_te)

vec_mfcc  = vec_df[[c for c in vec_df.columns if c.startswith('mfcc_norm_')]].values
vec_genre = vec_df[[c for c in vec_df.columns if c.startswith('genre_prob_')]].values
vec_tempo = vec_df[['tempo_norm']].values
print(f"  {len(feat_df)} songs loaded, {X_te.shape[0]} test samples\n")


# ── Helpers ────────────────────────────────────────────────────────────────────
def save(fig, name):
    path = os.path.join(OUT_DIR, name)
    fig.savefig(path, dpi=150, bbox_inches='tight', facecolor='white')
    plt.close(fig)
    print(f"  ✅  {name}")

def fp(qv, dm):
    return np.clip(1.0 - np.abs(dm - qv).mean(axis=1), 0, 1)

def to_mark(p):
    return 5 if p>=0.80 else 4 if p>=0.60 else 3 if p>=0.40 else 2 if p>=0.20 else 1

np.random.seed(42)
N_EVAL = 300
q_idx  = np.random.choice(len(vec_df), N_EVAL, replace=False)


# ════════════════════════════════════════════════════════════════════════════════
print("G1  — Genre distribution")
counts = feat_df['genre'].value_counts().reindex(GENRES)
fig, ax = plt.subplots(figsize=(10, 5))
bars = ax.bar(GENRES, counts.values,
              color=[GENRE_COLORS[g] for g in GENRES],
              width=0.6, edgecolor='white')
for bar, v in zip(bars, counts.values):
    ax.text(bar.get_x()+bar.get_width()/2, v+1,
            str(v), ha='center', va='bottom', fontsize=9)
ax.set_ylabel('Number of Songs', fontsize=11)
ax.set_xlabel('Genre', fontsize=11)
ax.set_ylim(0, 120)
ax.set_title('G1 — Genre Distribution in GTZAN Dataset',
             fontsize=13, fontweight='bold', pad=12)
plt.xticks(rotation=25, ha='right')
fig.tight_layout()
save(fig, 'G01_genre_distribution.png')


# ════════════════════════════════════════════════════════════════════════════════
print("G2  — Tempo box plot")
tempo_data = [feat_df[feat_df['genre']==g]['tempo'].values for g in GENRES]
fig, ax = plt.subplots(figsize=(11, 6))
bp = ax.boxplot(tempo_data, patch_artist=True,
                medianprops=dict(color='white', linewidth=2.5),
                whiskerprops=dict(linewidth=1.5),
                capprops=dict(linewidth=1.5),
                flierprops=dict(marker='o', markersize=3, alpha=0.4))
for patch, g in zip(bp['boxes'], GENRES):
    patch.set_facecolor(GENRE_COLORS[g]); patch.set_alpha(0.85)
ax.set_xticklabels(GENRES, rotation=25, ha='right', fontsize=10)
ax.set_ylabel('Tempo (BPM)', fontsize=11)
ax.set_title('G2 — Tempo Distribution by Genre',
             fontsize=13, fontweight='bold', pad=12)
fig.tight_layout()
save(fig, 'G02_tempo_boxplot.png')


# ════════════════════════════════════════════════════════════════════════════════
print("G3  — Confusion matrix")
cm   = confusion_matrix(y_te, y_pred)
cm_n = cm.astype(float) / cm.sum(axis=1, keepdims=True)
fig, ax = plt.subplots(figsize=(9, 7))
im = ax.imshow(cm_n, cmap='Blues', vmin=0, vmax=1)
plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04, label='Proportion')
for i in range(len(GENRES)):
    for j in range(len(GENRES)):
        col = 'white' if cm_n[i,j]>0.5 else 'black'
        ax.text(j, i, str(cm[i,j]), ha='center', va='center',
                fontsize=9, color=col, fontweight='bold')
ax.set_xticks(range(len(GENRES))); ax.set_yticks(range(len(GENRES)))
ax.set_xticklabels(GENRES, rotation=45, ha='right', fontsize=9)
ax.set_yticklabels(GENRES, fontsize=9)
ax.set_xlabel('Predicted Genre', fontsize=11)
ax.set_ylabel('True Genre', fontsize=11)
ax.set_title('G3 — Confusion Matrix (Genre Classification)',
             fontsize=13, fontweight='bold', pad=12)
ax.grid(False)
fig.tight_layout()
save(fig, 'G03_confusion_matrix.png')


# ════════════════════════════════════════════════════════════════════════════════
print("G4  — Precision / Recall / F1")
report = classification_report(y_te, y_pred,
                                target_names=GENRES, output_dict=True)
prec = [report[g]['precision'] for g in GENRES]
rec  = [report[g]['recall']    for g in GENRES]
f1   = [report[g]['f1-score']  for g in GENRES]
x = np.arange(len(GENRES)); w = 0.25
fig, ax = plt.subplots(figsize=(12, 5))
ax.bar(x-w, prec, w, label='Precision', color='#4472C4', alpha=0.9)
ax.bar(x,   rec,  w, label='Recall',    color='#ED7D31', alpha=0.9)
ax.bar(x+w, f1,   w, label='F1-Score',  color='#A9D18E', alpha=0.9)
ax.axhline(report['accuracy'], color='red', ls='--', lw=1.3,
           label=f"Overall accuracy ({report['accuracy']*100:.1f}%)")
ax.set_xticks(x)
ax.set_xticklabels(GENRES, rotation=25, ha='right', fontsize=9)
ax.set_ylim(0, 1.15)
ax.set_ylabel('Score', fontsize=11)
ax.set_title('G4 — Precision, Recall & F1-Score per Genre',
             fontsize=13, fontweight='bold', pad=12)
ax.legend(fontsize=9)
fig.tight_layout()
save(fig, 'G04_precision_recall_f1.png')


# ════════════════════════════════════════════════════════════════════════════════
print("G5  — ROC-AUC curves")
y_te_bin = label_binarize(y_te, classes=range(len(GENRES)))
fig, ax  = plt.subplots(figsize=(9, 7))
for i, g in enumerate(GENRES):
    fpr, tpr, _ = roc_curve(y_te_bin[:, i], y_pred_proba[:, i])
    ax.plot(fpr, tpr, lw=1.8, color=PALETTE[i],
            label=f'{g}  (AUC={auc(fpr,tpr):.2f})')
ax.plot([0,1],[0,1], 'k--', lw=1, alpha=0.4, label='Random baseline')
ax.set_xlabel('False Positive Rate', fontsize=11)
ax.set_ylabel('True Positive Rate',  fontsize=11)
ax.set_title('G5 — ROC Curves per Genre',
             fontsize=13, fontweight='bold', pad=12)
ax.legend(fontsize=8.5, loc='lower right')
fig.tight_layout()
save(fig, 'G05_roc_curves.png')


# ════════════════════════════════════════════════════════════════════════════════
print("G6  — Training loss curve")
iters = range(1, len(model.loss_curve_)+1)
fig, ax1 = plt.subplots(figsize=(9, 5))
ax2 = ax1.twinx()
ax1.plot(iters, model.loss_curve_,     color='#4472C4', lw=2.2, label='Training loss')
ax2.plot(iters, model.validation_scores_, color='#ED7D31', lw=2.2,
         ls='--', label='Validation accuracy')
ax1.set_xlabel('Epoch', fontsize=11)
ax1.set_ylabel('Loss',                fontsize=11, color='#4472C4')
ax2.set_ylabel('Validation Accuracy', fontsize=11, color='#ED7D31')
ax1.tick_params(axis='y', labelcolor='#4472C4')
ax2.tick_params(axis='y', labelcolor='#ED7D31')
lines  = ax1.get_lines() + ax2.get_lines()
labels = [l.get_label() for l in lines]
ax1.legend(lines, labels, fontsize=10, loc='center right')
ax1.set_title('G6 — MLP Training Loss & Validation Accuracy',
              fontsize=13, fontweight='bold', pad=12)
fig.tight_layout()
save(fig, 'G06_training_loss_curve.png')


# ════════════════════════════════════════════════════════════════════════════════
print("G7  — Strategy average rating")
strat_marks = {s: [] for s in STRATEGIES}
for idx in q_idx:
    mask = np.ones(len(vec_df), dtype=bool); mask[idx] = False
    pm = fp(vec_mfcc[idx],  vec_mfcc[mask])
    pg = fp(vec_genre[idx], vec_genre[mask])
    pt = fp(vec_tempo[idx], vec_tempo[mask])
    for s, (wm,wg,wt) in STRATEGIES.items():
        if s == 'Random':
            p = float(np.random.random())
        else:
            p = float(((pm*wm+pg*wg+pt*wt)/(wm+wg+wt)).max())
        strat_marks[s].append(to_mark(p))

avgs  = {s: np.mean(m) for s,m in strat_marks.items()}
names = list(avgs.keys())
vals  = list(avgs.values())
fig, ax = plt.subplots(figsize=(9, 5))
bars = ax.barh(names, vals, color=STRAT_COLORS, height=0.5, edgecolor='white')
for bar, v in zip(bars, vals):
    ax.text(v+0.05, bar.get_y()+bar.get_height()/2,
            f'{v:.2f}', va='center', fontsize=10, fontweight='bold')
ax.axvline(1.51, color='gray', ls='--', lw=1.2, alpha=0.7,
           label='Paper random baseline (1.51)')
ax.set_xlim(0, 5.8)
ax.set_xlabel('Average Rating (1–5)', fontsize=11)
ax.set_title('G7 — Average User Rating per Strategy',
             fontsize=13, fontweight='bold', pad=12)
ax.legend(fontsize=9)
fig.tight_layout()
save(fig, 'G07_strategy_avg_rating.png')


# ════════════════════════════════════════════════════════════════════════════════
print("G8  — KDE score distribution")
def get_scores(strategy):
    wm,wg,wt = STRATEGIES[strategy]
    np.random.seed(42)
    sc = []
    for idx in q_idx:
        mask = np.ones(len(vec_df), dtype=bool); mask[idx] = False
        pm = fp(vec_mfcc[idx],  vec_mfcc[mask])
        pg = fp(vec_genre[idx], vec_genre[mask])
        pt = fp(vec_tempo[idx], vec_tempo[mask])
        if strategy == 'Random':
            sc.append(np.random.random())
        else:
            sc.append(float(((pm*wm+pg*wg+pt*wt)/(wm+wg+wt)).max()))
    return sc

fig, ax = plt.subplots(figsize=(10, 5))
for s, col in zip(STRATEGIES.keys(), STRAT_COLORS):
    sc  = get_scores(s)
    kde = gaussian_kde(sc, bw_method=0.15)
    xs  = np.linspace(0, 1, 300)
    ax.plot(xs, kde(xs), lw=2.2, color=col, label=s)
    ax.fill_between(xs, kde(xs), alpha=0.07, color=col)
ax.set_xlabel('P_strategy Score', fontsize=11)
ax.set_ylabel('Density', fontsize=11)
ax.set_title('G8 — Score Distribution: All Strategies vs Random',
             fontsize=13, fontweight='bold', pad=12)
ax.legend(fontsize=9)
fig.tight_layout()
save(fig, 'G08_kde_score_distribution.png')


# ════════════════════════════════════════════════════════════════════════════════
print("G9  — Same-genre hit rate")
def same_genre_rate(strategy):
    wm,wg,wt = STRATEGIES[strategy]
    hits = 0
    np.random.seed(42)
    for idx in q_idx:
        mask = np.ones(len(vec_df), dtype=bool); mask[idx] = False
        pm = fp(vec_mfcc[idx],  vec_mfcc[mask])
        pg = fp(vec_genre[idx], vec_genre[mask])
        pt = fp(vec_tempo[idx], vec_tempo[mask])
        sub = vec_df[mask].reset_index(drop=True)
        if strategy == 'Random':
            top_i = np.random.randint(0, mask.sum())
        else:
            top_i = int(((pm*wm+pg*wg+pt*wt)/(wm+wg+wt)).argmax())
        if sub.iloc[top_i]['genre'] == vec_df.iloc[idx]['genre']:
            hits += 1
    return hits / N_EVAL * 100

rates = {s: same_genre_rate(s) for s in STRATEGIES}
fig, ax = plt.subplots(figsize=(9, 5))
bars = ax.bar(list(rates.keys()), list(rates.values()),
              color=STRAT_COLORS, width=0.5, edgecolor='white')
for bar, v in zip(bars, rates.values()):
    ax.text(bar.get_x()+bar.get_width()/2, v+1,
            f'{v:.1f}%', ha='center', fontsize=10, fontweight='bold')
ax.axhline(10, color='gray', ls='--', lw=1.2, alpha=0.7,
           label='Random baseline (10%)')
ax.set_ylim(0, 115)
ax.set_ylabel('Same-Genre Hit Rate (%)', fontsize=11)
ax.set_title('G9 — Same-Genre Recommendation Hit Rate per Strategy',
             fontsize=13, fontweight='bold', pad=12)
ax.legend(fontsize=9)
plt.xticks(rotation=15, ha='right')
fig.tight_layout()
save(fig, 'G09_same_genre_hit_rate.png')


# ════════════════════════════════════════════════════════════════════════════════
print("G10 — MFCC heatmap")
mean_c = [f'mfcc{i+1}_mean' for i in range(13)]
mg     = feat_df.groupby('genre')[mean_c].mean()
mg.columns = [f'MFCC {i+1}' for i in range(13)]
dz     = (mg - mg.mean()) / mg.std()
fig, ax = plt.subplots(figsize=(13, 6))
im = ax.imshow(dz.values, cmap='RdYlBu_r', aspect='auto')
plt.colorbar(im, ax=ax, fraction=0.025, pad=0.02, label='Z-score')
ax.set_xticks(range(13))
ax.set_xticklabels([f'MFCC {i+1}' for i in range(13)],
                   rotation=45, ha='right', fontsize=9)
ax.set_yticks(range(len(GENRES)))
ax.set_yticklabels(mg.index, fontsize=10)
for i in range(len(GENRES)):
    for j in range(13):
        v   = dz.values[i,j]
        col = 'white' if abs(v)>1.2 else 'black'
        ax.text(j, i, f'{v:.1f}', ha='center', va='center',
                fontsize=7, color=col)
ax.set_title('G10 — Mean MFCC Coefficients by Genre (Z-scored)',
             fontsize=13, fontweight='bold', pad=12)
ax.grid(False)
fig.tight_layout()
save(fig, 'G10_mfcc_heatmap.png')


# ════════════════════════════════════════════════════════════════════════════════
print("G11 — t-SNE  (may take ~60 seconds) …")
combined = np.hstack([vec_mfcc, vec_genre, vec_tempo])
emb = TSNE(n_components=2, perplexity=40, random_state=42,
           n_iter=1000, learning_rate='auto', init='pca').fit_transform(combined)
fig, ax = plt.subplots(figsize=(10, 8))
for i, g in enumerate(GENRES):
    mask = vec_df['genre'].values == g
    ax.scatter(emb[mask,0], emb[mask,1],
               c=PALETTE[i], s=18, alpha=0.75, label=g, edgecolors='none')
ax.set_title('G11 — t-SNE Visualization of Song Feature Space',
             fontsize=13, fontweight='bold', pad=12)
ax.legend(fontsize=9, markerscale=2, loc='upper right')
ax.set_xlabel('t-SNE Dimension 1', fontsize=10)
ax.set_ylabel('t-SNE Dimension 2', fontsize=10)
fig.tight_layout()
save(fig, 'G11_tsne_clusters.png')


# ════════════════════════════════════════════════════════════════════════════════
print("G12 — Radar chart")
gp_cols = [c for c in vec_df.columns if c.startswith('genre_prob_')]
angles  = np.linspace(0, 2*np.pi, len(GENRES), endpoint=False).tolist()
angles += angles[:1]
fig, axes = plt.subplots(2, 5, figsize=(18, 7),
                          subplot_kw=dict(polar=True))
fig.suptitle('G12 — Genre Probability Vector (Radar) — One Sample per Genre',
             fontsize=13, fontweight='bold', y=1.01)
for ax, g in zip(axes.flatten(), GENRES):
    idx   = vec_df[vec_df['genre']==g].index[0]
    probs = vec_df.loc[idx, gp_cols].values.tolist() + [vec_df.loc[idx, gp_cols].values[0]]
    col   = GENRE_COLORS[g]
    ax.plot(angles, probs, color=col, lw=2)
    ax.fill(angles, probs, color=col, alpha=0.2)
    ax.set_xticks(angles[:-1])
    ax.set_xticklabels([g[:3] for g in GENRES], fontsize=7)
    ax.set_ylim(0, 1); ax.set_yticks([0.25,0.5,0.75])
    ax.set_yticklabels([], fontsize=6)
    ax.set_title(g.capitalize(), fontsize=10, fontweight='bold',
                 color=col, pad=10)
    ax.grid(color='grey', alpha=0.3)
fig.tight_layout()
save(fig, 'G12_radar_genre_probability.png')

print(f"\n{'='*52}\n  All 12 graphs saved to: {OUT_DIR}\n{'='*52}")
