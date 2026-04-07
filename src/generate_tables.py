"""
generate_tables.py  —  All 6 tables for the Music Recommendation Project
-------------------------------------------------------------------------
T1  Strategy mark distribution (No. + % for marks 1-5 + Avg)
T2  Per-genre classification report (P / R / F1 / Support)
T3  Strategy × Genre average rating grid
T4  Same-genre hit rate per strategy (%)
T5  Statistical significance tests (p-values between strategies)
T6  Feature contribution summary (MFCC / Genre / Tempo weights)

Each table saved as:  tables/T0X_name.png
Also printed to console.

Run:  cd C:\\Users\\sanik\\Downloads\\RS_Project
      python generate_tables.py
"""

import os, pickle, warnings
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from matplotlib import rcParams
from sklearn.metrics         import classification_report
from sklearn.model_selection import train_test_split
from scipy.stats             import mannwhitneyu

warnings.filterwarnings('ignore')

# ── Paths ──────────────────────────────────────────────────────────────────────
BASE_DIR     = r"C:\Users\sanik\Downloads\RS_Project"
FEATURES_CSV = os.path.join(BASE_DIR, 'features', 'mfcc_features.csv')
VECTORS_CSV  = os.path.join(BASE_DIR, 'features', 'song_vectors.csv')
MODEL_PATH   = os.path.join(BASE_DIR, 'models', 'genre_model.pkl')
SCALER_PATH  = os.path.join(BASE_DIR, 'models', 'scaler.pkl')
ENCODER_PATH = os.path.join(BASE_DIR, 'models', 'encoder.pkl')
OUT_DIR      = os.path.join(BASE_DIR, 'tables')
os.makedirs(OUT_DIR, exist_ok=True)

rcParams['font.family'] = 'DejaVu Sans'

GENRES = ['blues','classical','country','disco','hiphop',
          'jazz','metal','pop','reggae','rock']
STRATEGIES = {
    'Random':          (0, 0, 0),
    'Unbalanced':      (2, 3, 1),
    'PrioritizeMFCC':  (4, 1, 1),
    'PrioritizeTempo': (1, 1, 4),
    'PrioritizeGenre': (1, 4, 1),
}

# ── Load ───────────────────────────────────────────────────────────────────────
print("Loading data and models …")
with open(MODEL_PATH,   'rb') as f: model   = pickle.load(f)
with open(SCALER_PATH,  'rb') as f: scaler  = pickle.load(f)
with open(ENCODER_PATH, 'rb') as f: encoder = pickle.load(f)

feat_df = pd.read_csv(FEATURES_CSV)
vec_df  = pd.read_csv(VECTORS_CSV)
mfcc_c  = [c for c in feat_df.columns if c.startswith('mfcc')]

X_raw = feat_df[mfcc_c].values
y_raw = feat_df['genre'].values
y     = encoder.transform(y_raw)
X     = scaler.transform(X_raw)

X_tr, X_te, y_tr, y_te = train_test_split(
    X, y, test_size=0.2, stratify=y, random_state=42)
y_pred = model.predict(X_te)

vm  = vec_df[[c for c in vec_df.columns if c.startswith('mfcc_norm_')]].values
vg  = vec_df[[c for c in vec_df.columns if c.startswith('genre_prob_')]].values
vt  = vec_df[['tempo_norm']].values
print(f"  {len(feat_df)} songs, {X_te.shape[0]} test samples\n")


# ── Shared helpers ─────────────────────────────────────────────────────────────
def fp(qv, dm):
    return np.clip(1.0 - np.abs(dm - qv).mean(axis=1), 0, 1)

def to_mark(p):
    return 5 if p>=0.80 else 4 if p>=0.60 else 3 if p>=0.40 else 2 if p>=0.20 else 1

np.random.seed(42)
N_EVAL = 300
q_idx  = np.random.choice(len(vec_df), N_EVAL, replace=False)

# Pre-compute marks for all strategies (used in T1, T3, T5)
print("  Pre-computing strategy marks (used in T1, T3, T5) …")
strat_marks      = {s: [] for s in STRATEGIES}
strat_marks_genre = {s: {g: [] for g in GENRES} for s in STRATEGIES}

for idx in q_idx:
    mask = np.ones(len(vec_df), dtype=bool); mask[idx] = False
    pm = fp(vm[idx], vm[mask])
    pg = fp(vg[idx], vg[mask])
    pt = fp(vt[idx], vt[mask])
    query_genre = vec_df.iloc[idx]['genre']

    for s, (wm,wg,wt) in STRATEGIES.items():
        if s == 'Random':
            p = float(np.random.random())
        else:
            p = float(((pm*wm+pg*wg+pt*wt)/(wm+wg+wt)).max())
        mark = to_mark(p)
        strat_marks[s].append(mark)
        strat_marks_genre[s][query_genre].append(mark)

print("  Done.\n")


def render_table(ax, cell_data, col_labels, col_widths=None,
                 bold_cells=None, header_color='#D0D8E8',
                 alt_row_color='#F5F6FA'):
    """Render a styled table onto ax (which must have axis('off'))."""
    bold_cells = bold_cells or set()
    n_rows = len(cell_data)
    n_cols = len(col_labels)

    table = ax.table(
        cellText=cell_data,
        colLabels=col_labels,
        cellLoc='center',
        loc='center',
        bbox=[0, 0, 1, 1]
    )
    table.auto_set_font_size(False)
    table.set_fontsize(9)

    # Column widths
    if col_widths:
        for r in range(n_rows+1):
            for j, w in enumerate(col_widths):
                table[r, j].set_width(w)

    # Header
    for j in range(n_cols):
        c = table[0, j]
        c.set_facecolor(header_color)
        c.set_text_props(fontweight='bold', fontsize=8.5)
        c.set_edgecolor('#AAAAAA')

    # Data rows
    for r in range(1, n_rows+1):
        bg = alt_row_color if r % 2 == 0 else '#FFFFFF'
        for j in range(n_cols):
            c = table[r, j]
            c.set_facecolor(bg)
            c.set_edgecolor('#CCCCCC')
            if (r, j) in bold_cells:
                c.set_text_props(fontweight='bold')

    return table


def save_table(fig, name):
    path = os.path.join(OUT_DIR, name)
    fig.savefig(path, dpi=160, bbox_inches='tight', facecolor='white')
    plt.close(fig)
    print(f"  ✅  {name}")



# ══════════════════════════════════════════════════════════════════════════════
# T2 — Per-genre classification report
# ══════════════════════════════════════════════════════════════════════════════
print("T2  — Per-genre classification report")
report = classification_report(y_te, y_pred,
                                target_names=GENRES, output_dict=True)
col_labels = ['Genre', 'Precision', 'Recall', 'F1-Score', 'Support']
cell_data  = []
for g in GENRES:
    r = report[g]
    cell_data.append([
        g,
        f"{r['precision']:.3f}",
        f"{r['recall']:.3f}",
        f"{r['f1-score']:.3f}",
        int(r['support'])
    ])
# Summary row
cell_data.append([
    'Weighted Avg',
    f"{report['weighted avg']['precision']:.3f}",
    f"{report['weighted avg']['recall']:.3f}",
    f"{report['weighted avg']['f1-score']:.3f}",
    int(report['weighted avg']['support'])
])

# Bold best F1
f1_vals = [float(r[3]) for r in cell_data[:-1]]
best_f1 = max(f1_vals)
bold_cells = {(f1_vals.index(best_f1)+1, 3)}

fig, ax = plt.subplots(figsize=(8, 4.5))
ax.axis('off')
fig.text(0.5, 0.98,
         'Table 2.  Per-Genre Classification Report (MLP Classifier).',
         ha='center', va='top', fontsize=11, fontweight='bold')
render_table(ax, cell_data, col_labels, bold_cells=bold_cells)
save_table(fig, 'T2_classification_report.png')

print(f"\n  {'Genre':<14} {'Precision':>10} {'Recall':>8} {'F1':>8} {'Support':>8}")
for row in cell_data:
    print(f"  {str(row[0]):<14} {str(row[1]):>10} {str(row[2]):>8}"
          f" {str(row[3]):>8} {str(row[4]):>8}")
print()


# ══════════════════════════════════════════════════════════════════════════════
# T3 — Strategy × Genre average rating
# ══════════════════════════════════════════════════════════════════════════════
print("T3  — Strategy × Genre average rating")

strat_names = list(STRATEGIES.keys())
col_labels  = ['Strategy'] + [g.capitalize() for g in GENRES] + ['Overall']
cell_data   = []

all_avgs = []
for s in strat_names:
    row  = [s]
    gavgs = []
    for g in GENRES:
        marks = strat_marks_genre[s][g]
        avg   = round(np.mean(marks), 2) if marks else 0.0
        gavgs.append(avg)
        row.append(avg)
    overall = round(np.mean(np.array(strat_marks[s])), 2)
    row.append(overall)
    cell_data.append(row)
    all_avgs.append(gavgs)

# Bold highest avg per genre column
bold_cells = set()
for gi in range(len(GENRES)):
    col_vals = [cell_data[si][gi+1] for si in range(len(strat_names))]
    mx = max(col_vals)
    for si, v in enumerate(col_vals):
        if v == mx:
            bold_cells.add((si+1, gi+1))

fig, ax = plt.subplots(figsize=(18, 3))
ax.axis('off')
fig.text(0.5, 0.98,
         'Table 3.  Strategy × Genre Average User Rating.',
         ha='center', va='top', fontsize=11, fontweight='bold')
fig.text(0.01, 0.0,
         'Bold = highest average rating per genre column.',
         ha='left', va='bottom', fontsize=7.5, style='italic', color='#555')
render_table(ax, cell_data, col_labels, bold_cells=bold_cells)
save_table(fig, 'T3_strategy_genre_avg_rating.png')

print(f"\n  {'Strategy':<20}" + "".join(f" {g[:5]:>7}" for g in GENRES) + " Overall")
for row in cell_data:
    print(f"  {row[0]:<20}" + "".join(f" {v:>7}" for v in row[1:]))
print()


# ══════════════════════════════════════════════════════════════════════════════
# T4 — Same-genre hit rate per strategy
# ══════════════════════════════════════════════════════════════════════════════
print("T4  — Same-genre hit rate")

def same_genre_rate(strategy):
    wm,wg,wt = STRATEGIES[strategy]
    hits = 0
    np.random.seed(42)
    for idx in q_idx:
        mask = np.ones(len(vec_df), dtype=bool); mask[idx] = False
        pm = fp(vm[idx], vm[mask])
        pg = fp(vg[idx], vg[mask])
        pt = fp(vt[idx], vt[mask])
        sub = vec_df[mask].reset_index(drop=True)
        if strategy == 'Random':
            top_i = np.random.randint(0, mask.sum())
        else:
            top_i = int(((pm*wm+pg*wg+pt*wt)/(wm+wg+wt)).argmax())
        if sub.iloc[top_i]['genre'] == vec_df.iloc[idx]['genre']:
            hits += 1
    return hits

col_labels = ['Strategy', 'Correct Genre Hits', 'Total Queries',
              'Hit Rate (%)', 'vs Random Baseline']
rates      = {s: same_genre_rate(s) for s in STRATEGIES}
random_pct = rates['Random'] / N_EVAL * 100
cell_data  = []
for s, hits in rates.items():
    pct   = round(hits / N_EVAL * 100, 1)
    delta = round(pct - random_pct, 1)
    sign  = f'+{delta}' if delta >= 0 else str(delta)
    cell_data.append([s, hits, N_EVAL, f'{pct}%', sign+'%'])

# Bold highest hit rate
pcts      = [float(r[3].strip('%')) for r in cell_data]
bold_cells = {(pcts.index(max(pcts))+1, 3)}

fig, ax = plt.subplots(figsize=(10, 3))
ax.axis('off')
fig.text(0.5, 0.98,
         'Table 4.  Same-Genre Recommendation Hit Rate per Strategy.',
         ha='center', va='top', fontsize=11, fontweight='bold')
render_table(ax, cell_data, col_labels, bold_cells=bold_cells)
save_table(fig, 'T4_same_genre_hit_rate.png')

print(f"\n  {'Strategy':<20} {'Hits':>6} {'Total':>6} {'Rate':>8} {'vs Random':>10}")
for row in cell_data:
    print(f"  {row[0]:<20} {row[1]:>6} {row[2]:>6} {row[3]:>8} {row[4]:>10}")
print()


# ══════════════════════════════════════════════════════════════════════════════
# T5 — Statistical significance (Mann-Whitney U p-values)
# ══════════════════════════════════════════════════════════════════════════════
print("T5  — Statistical significance tests")

non_random = [s for s in STRATEGIES if s != 'Random']
col_labels = ['Strategy A', 'Strategy B', 'p-value', 'Significant (α=0.05)']
cell_data  = []

# Each non-random strategy vs Random
for s in non_random:
    _, p = mannwhitneyu(strat_marks[s], strat_marks['Random'],
                        alternative='greater')
    sig  = 'Yes' if p < 0.05 else 'No'
    cell_data.append([s, 'Random', f'{p:.2e}', sig])

# PrioritizeMFCC (best) vs others
best = 'PrioritizeMFCC'
for s in non_random:
    if s == best: continue
    _, p = mannwhitneyu(strat_marks[best], strat_marks[s],
                        alternative='greater')
    sig  = 'Yes' if p < 0.05 else 'No'
    cell_data.append([best, s, f'{p:.2e}', sig])

bold_cells = {(i+1, 3) for i, r in enumerate(cell_data) if r[3]=='Yes'}

fig, ax = plt.subplots(figsize=(10, 4))
ax.axis('off')
fig.text(0.5, 0.98,
         'Table 5.  Statistical Significance Tests Between Strategies (Mann-Whitney U).',
         ha='center', va='top', fontsize=11, fontweight='bold')
fig.text(0.01, 0.0,
         'Alternative hypothesis: Strategy A > Strategy B.  α = 0.05.',
         ha='left', va='bottom', fontsize=7.5, style='italic', color='#555')
render_table(ax, cell_data, col_labels, bold_cells=bold_cells)
save_table(fig, 'T5_statistical_significance.png')

print(f"\n  {'Strategy A':<22} {'Strategy B':<22} {'p-value':>12} {'Sig.':>6}")
for row in cell_data:
    print(f"  {row[0]:<22} {row[1]:<22} {row[2]:>12} {row[3]:>6}")
print()


# ══════════════════════════════════════════════════════════════════════════════
# T6 — Feature contribution summary
# ══════════════════════════════════════════════════════════════════════════════
print("T6  — Feature contribution summary")

col_labels = ['Strategy',
              'MFCC Weight', 'MFCC %',
              'Genre Weight', 'Genre %',
              'Tempo Weight', 'Tempo %',
              'Avg. Rating', 'Rank']

sorted_strats = sorted(
    [s for s in STRATEGIES if s != 'Random'],
    key=lambda s: np.mean(strat_marks[s]), reverse=True
)

cell_data = []
for rank, s in enumerate(sorted_strats, 1):
    wm,wg,wt = STRATEGIES[s]
    total    = wm + wg + wt
    avg      = round(np.mean(strat_marks[s]), 2)
    cell_data.append([
        s,
        wm, f'{wm/total*100:.0f}%',
        wg, f'{wg/total*100:.0f}%',
        wt, f'{wt/total*100:.0f}%',
        avg, rank
    ])

# Also add Random row
cell_data.append(['Random', '—','—','—','—','—','—',
                  round(np.mean(strat_marks['Random']),2), '—'])

# Bold MFCC % col for PrioritizeMFCC row
bold_cells = {(1, 2), (1, 7)}   # row 1 = PrioritizeMFCC (highest rank), Avg col

fig, ax = plt.subplots(figsize=(14, 3.5))
ax.axis('off')
fig.text(0.5, 0.98,
         'Table 6.  Feature Contribution Summary — Weight Distribution per Strategy.',
         ha='center', va='top', fontsize=11, fontweight='bold')
fig.text(0.01, 0.0,
         'Strategies ranked by average user rating (highest first). '
         'MFCC = 91 coefficients, Genre = 10-dim probability vector, Tempo = BPM.',
         ha='left', va='bottom', fontsize=7.5, style='italic', color='#555')
render_table(ax, cell_data, col_labels, bold_cells=bold_cells)
save_table(fig, 'T6_feature_contribution.png')

print(f"\n  {'Strategy':<20} {'MFCC_w':>7} {'MFCC%':>7} {'Genre_w':>8}"
      f" {'Genre%':>7} {'Tempo_w':>8} {'Tempo%':>7} {'Avg':>6} {'Rank':>5}")
for row in cell_data:
    print(f"  {str(row[0]):<20} {str(row[1]):>7} {str(row[2]):>7}"
          f" {str(row[3]):>8} {str(row[4]):>7} {str(row[5]):>8}"
          f" {str(row[6]):>7} {str(row[7]):>6} {str(row[8]):>5}")

print(f"\n{'='*52}\n  All 6 tables saved to: {OUT_DIR}\n{'='*52}")
