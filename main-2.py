"""
File: main-2.py
Author: Chuncheng Zhang
Date: 2026-01-19
Copyright & Email: chuncheng.zhang@ia.ac.cn

Purpose:
    Analysis erds with Clustering Statistical.

Functions:
    1. Requirements and constants
    2. Function and class
    3. Play ground
    4. Pending
    5. Pending
"""


# %% ---- 2026-01-19 ------------------------
# Requirements and constants
from itertools import combinations
from scipy import stats
from collections import defaultdict
import numpy as np
import joblib
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

from rich import print
from pathlib import Path
from tqdm.auto import tqdm
from itertools import product

from util.anova import analysis_anova, after_anova_cluster_test
from util.ttest import ttest

# %%
DATA_DIR = Path('./data/erd.exampleChannels.detail')
MODES = ['meg', 'eeg']
CHANNELS = ['0-C3', '0-MLC42']
EVENTS = ['0', '1', '2', '3']
BANDS = ['alpha', 'beta']

OUTPUT_DIR = Path('img-v2')
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# %% ---- 2026-01-19 ------------------------
# Function and class


# %% ---- 2026-01-19 ------------------------
# Play ground

dfs = []
for evt, mode, channel in tqdm(product(EVENTS, MODES, CHANNELS)):
    fpath = DATA_DIR / f'{mode}-{evt}-{channel}.dump'
    if not fpath.is_file():
        continue
    obj = joblib.load(fpath)
    mat = obj['mat']
    freqs = obj['freqs']
    times = obj['times']

    df = pd.DataFrame(mat, index=freqs, columns=times)
    # 使用 melt 重塑为三列
    df = df.reset_index().melt(
        id_vars='index',
        var_name='times',
        value_name='values'
    )
    df = df.rename(columns={'index': 'freqs'})
    df['mode'] = mode
    df['evt'] = evt
    dfs.append(df)

table = pd.concat(dfs)
table['values'] *= 3
print(table)

# 典型脑电波频率范围（单位：Hz）
bands = {
    'delta': (0.5, 4),
    'theta': (4, 8),
    'alpha': (8, 13),
    'beta': (13, 30),  # (20, 30),
    'gamma': (30, 45)
}

table['band'] = table['freqs'].map(
    lambda f: [k for k, v in bands.items() if f >= v[0] and f <= v[1]][0])
table = table[table['band'].isin(BANDS)]
print(table)

# %%
idx = (
    table
    .groupby(["times", "mode", "evt", "band"])["values"]
    .idxmin()
)

table = table.loc[idx].reset_index(drop=True)
print(table)


# %%
times = sorted(table['times'].unique())
results = []
for mode in tqdm(MODES, 'Modes'):
    for band in tqdm(BANDS, 'Bands'):
        sub_table = table.query(f'mode=="{mode}" and band=="{band}"').copy()
        for t in times:
            _df = sub_table.query(f'times=={t}')
            _anova_table = analysis_anova(_df)
            results.append({
                'mode': mode,
                'band': band,
                'time': t,
                'F': _anova_table.loc['C(evt)', 'F'],
                'p': _anova_table.loc['C(evt)', 'PR(>F)'],
            })

results = pd.DataFrame(results)
print(results)

# %% ---- 2026-01-19 ------------------------
# Pending
df = results.query('time>0 & time<4').copy()
# df = results.copy()
cluster_results = after_anova_cluster_test(df)
# print(cluster_results)

df_significant = df[df['p'] < 0.05]

g = sns.FacetGrid(
    df,
    col='band',
    row='mode',
    margin_titles=True,
    height=4,
    aspect=2
)

g.map(sns.scatterplot, 'time', 'F')

plt.show()

# %%
times = list(cluster_results.values())[0]['times']
print(times, len(times))

# %% ---- 2026-01-19 ------------------------
# Pending
timelines = {}

for k, v in cluster_results.items():
    print(f'Cluster {k}:')
    p_values = np.array(v['cluster_p_values'])
    clusters = [v['clusters'][i] for i, p in enumerate(p_values) if p < 0.05]
    p_values = p_values[p_values < 0.05]
    tl = np.array(times) * 0
    for c in clusters:
        tl[c] = 1
    timelines[k] = tl


# %%
index = ['-'.join(k) for k in timelines.keys()]
columns = pd.Index(times, name='time')
df = pd.DataFrame(timelines.values(), index=index, columns=columns)
df.to_csv(OUTPUT_DIR / 'Significant clusters timelines.csv')

# %%

fig, axes = plt.subplots(2, 2, figsize=(12, 12), dpi=600)
for i, band in enumerate(tqdm(['alpha', 'beta'], 'bands')):
    for j, mode in enumerate(tqdm(['eeg', 'meg'], 'modes')):
        ax = axes[j, i]
        key = (mode, band)
        ax.plot(times, timelines[key], label=f'{mode}_{band}')
        ax.set_title(f'{mode} - {band}')
        ax.set_xlabel('Time (s)')
        ax.set_ylabel('Significant Cluster')
        ax.legend()
plt.savefig(OUTPUT_DIR / 'Significant clusters timelines.png')
plt.show()

# %%
results_df = ttest(table.copy())
print(results_df)

# %%
indexes = []
for i, row in tqdm(results_df.iterrows()):
    if row['p_value'] > 0.05:
        continue
    mode = row['mode']
    band = row['band']
    t = row['time']
    _times = timelines[(mode, band)]
    _times = [times[i] for i, v in enumerate(_times) if v > 0.5]
    if t in _times:
        indexes.append(row.name)
df = results_df.loc[indexes]
df['compare'] = df['evt1'].astype(str) + ' vs ' + df['evt2'].astype(str)

g = sns.FacetGrid(
    df,
    col='band',
    row='mode',
    hue='compare',
    margin_titles=True,
    height=4,
    aspect=2,
    xlim=[0, 4]
)

g.map(sns.scatterplot, 'time', 'compare')

df.to_csv(OUTPUT_DIR / 'Significant T-test points.csv', index=False)

plt.savefig(OUTPUT_DIR / 'Significant T-test points.png')
plt.show()

print(df, len(df))


# %%

# %%
