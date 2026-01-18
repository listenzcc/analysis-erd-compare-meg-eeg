"""
File: main.py
Author: Chuncheng Zhang
Date: 2026-01-18
Copyright & Email: chuncheng.zhang@ia.ac.cn

Purpose:
    Analysis the erd of example channels

Functions:
    1. Requirements and constants
    2. Function and class
    3. Play ground
    4. Pending
    5. Pending
"""


# %% ---- 2026-01-18 ------------------------
# Requirements and constants
import joblib
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

from pathlib import Path
from tqdm.auto import tqdm
from itertools import product

# %%
DATA_DIR = Path('./data/erd.exampleChannels.detail')
MODES = ['meg', 'eeg']
CHANNELS = ['0-C3', '0-MLC42']
EVENTS = ['0', '1', '2', '3']

OUTPUT_DIR = Path('img')
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# %% ---- 2026-01-18 ------------------------
# Function and class


# %% ---- 2026-01-18 ------------------------
# Play ground
mat0_meg = joblib.load(DATA_DIR / f'meg-4-0-MLC42.dump')['mat']
mat0_eeg = joblib.load(DATA_DIR / f'eeg-4-0-C3.dump')['mat']

dfs = []
for evt, mode, channel in tqdm(product(EVENTS, MODES, CHANNELS)):
    fpath = DATA_DIR / f'{mode}-{evt}-{channel}.dump'
    if not fpath.is_file():
        continue
    obj = joblib.load(fpath)
    mat = obj['mat']
    freqs = obj['freqs']
    times = obj['times']

    if mode == 'meg':
        mat -= mat0_meg
    elif mode == 'eeg':
        mat -= mat0_eeg
    else:
        raise ValueError(f'Incorrect {mode=}')

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

# %%
# meg = table.query(f'mode=="meg"').copy()
# eeg = table.query(f'mode=="eeg"').copy()

# print(meg)
# print(eeg)

# %%
sns.set_theme(context='paper', style='ticks')

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
print(table)

# %% ---- 2026-01-18 ------------------------
# Pending


fig, axes = plt.subplots(2, 2, figsize=(12, 12), dpi=600)

for i, band in enumerate(tqdm(['alpha', 'beta'], 'bands')):
    for j, mode in enumerate(tqdm(['meg', 'eeg'], 'modes')):
        ax = axes[i][j]
        l, h = bands[band]
        df = table.query(f'mode=="{mode}" & band=="{band}"')
        sns.lineplot(df, x='times', y='values', hue='evt', ax=ax)
        ax.set_title(f'{mode} - {band}')

fig.tight_layout()
fig.savefig(OUTPUT_DIR / 'Time series of bands.png')
plt.show()

# %%
fig, axes = plt.subplots(1, 2, figsize=(12, 6), dpi=600)

for j, mode in enumerate(tqdm(['meg', 'eeg'])):
    ax = axes[j]
    l, h = bands[band]
    df = table.query(f'mode=="{mode}" & times>0 & times<4')
    sns.lineplot(df, x='freqs', y='values', hue='evt', ax=ax)
    ax.set_title(f'{mode}')

fig.tight_layout()
fig.savefig(OUTPUT_DIR / 'Freqs comparison.png')
plt.show()

# %%
fig, axes = plt.subplots(1, 2, figsize=(12, 6), dpi=600)
for j, mode in enumerate(tqdm(['meg', 'eeg'])):
    ax = axes[j]
    l, h = bands[band]
    df = table.query(f'mode=="{mode}" & times>0 & times<4')
    sns.barplot(df, x='band', y='values', hue='evt', ax=ax)
    ax.set_title(f'{mode}')

fig.tight_layout()
fig.savefig(OUTPUT_DIR / 'Bar comparison.png')
plt.show()

# %% ---- 2026-01-18 ------------------------
# Pending
