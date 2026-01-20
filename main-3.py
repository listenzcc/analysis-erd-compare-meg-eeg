"""
File: main-3.py
Author: Chuncheng Zhang
Date: 2026-01-20
Copyright & Email: chuncheng.zhang@ia.ac.cn

Purpose:
    Analysis data with Clustering and t-test

Functions:
    1. Requirements and constants
    2. Function and class
    3. Play ground
    4. Pending
    5. Pending
"""


# %% ---- 2026-01-20 ------------------------
# Requirements and constants
import json
from collections import defaultdict
from itertools import combinations
from scipy import stats
import numpy as np
import joblib
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

from pathlib import Path
from tqdm.auto import tqdm
from itertools import product

from utils.anova import anova_analysis, anova_cluster_permutation_test

# %%
DATA_DIR = Path('./data/erd.exampleChannels.detail')
MODES = ['meg', 'eeg']
CHANNELS = ['0-C3', '0-MLC42']
EVENTS = ['0', '1', '2', '3']
BANDS = ['alpha', 'beta']

OUTPUT_DIR = Path('img-v3')
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# %% ---- 2026-01-20 ------------------------
# Function and class

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

# %%
idx = (
    table
    .groupby(["times", "mode", "evt", "band"])["values"]
    .idxmin()
)

table = table.loc[idx].reset_index(drop=True)
print(table)
table.to_csv(OUTPUT_DIR / 'erd example channels summary.csv', index=False)

# %%
table = table[['times', 'values', 'mode', 'evt', 'band']]
table = table[table['band'].isin(BANDS)]
print(table)

# %%
# table = table[table['times'].between(0, 4.0)]
# print(table)

# %%
times = np.array(sorted(table['times'].unique()))
times

# %% ---- 2026-01-20 ------------------------
# Play ground
anova_timelines = {}
for mode in MODES:
    for band in BANDS:
        anova_timelines[(mode, band)] = times * 0

anova_results = []

for t in tqdm(times):
    for mode in MODES:
        for band in BANDS:
            mode_band_df = table[(table['mode'] == mode) & (
                table['band'] == band) & (table['times'] == t)]
            anova_table = anova_analysis(mode_band_df)
            anova_timelines[(mode, band)][times ==
                                          t] = anova_table.loc['C(evt)', 'PR(>F)']
            anova_results.append({
                'time': t,
                'mode': mode,
                'band': band,
                'p': anova_table.loc['C(evt)', 'PR(>F)'],
                'F': anova_table.loc['C(evt)', 'F']
            })

anova_results_df = pd.DataFrame(anova_results)

print(anova_timelines)
print(anova_results_df)

# %%
for k, v in anova_timelines.items():
    mode, band = k
    plt.figure(figsize=(10, 4))
    plt.plot(times, v, label=f'ANOVA p-values ({mode}, {band})')
    plt.axhline(0.05, color='red', linestyle='--',
                label='Significance Level (0.05)')
    plt.xlabel('Time (ms)')
    plt.ylabel('p-value')
    plt.title(
        f'ANOVA p-values over Time for {mode.upper()} - {band.capitalize()} Band')
    plt.yscale('log')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / f'anova_pvalues_{mode}_{band}.png')
    plt.show()

# %% ---- 2026-01-20 ------------------------
# Pending
cluster_results = anova_cluster_permutation_test(anova_results_df)

# %%
print(cluster_results)

# %%
cluster_timelines = {}
for k, v in cluster_results.items():
    mode, band = k
    tl = times * 0
    for c in v['clusters']:
        tl[c] = 1
    cluster_timelines[(mode, band)] = tl

print(cluster_timelines)

for k, v in cluster_timelines.items():
    mode, band = k
    plt.figure(figsize=(10, 4))
    plt.plot(times, v, label=f'Cluster Significance ({mode}, {band})')
    plt.xlabel('Time (ms)')
    plt.ylabel('Significant Cluster (1=Yes, 0=No)')
    plt.title(
        f'Cluster Significance over Time for {mode.upper()} - {band.capitalize()} Band')
    plt.ylim(-0.1, 1.1)
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / f'cluster_significance_{mode}_{band}.png')
    plt.show()

cluster_timelines_df = pd.DataFrame(list(cluster_timelines.values()), index=list(
    cluster_timelines.keys()), columns=times)
cluster_timelines_df.to_csv(OUTPUT_DIR / 'anova_cluster_timelines.csv')

# %%
# ! t-test

# 假设您的数据已经在df中
# 如果还没有读取数据，请先读取：
# df = pd.read_csv('your_data.csv')

# 创建存储结果的列表
results = []
df = table.copy()

# 按band和mode分组
for (band, mode), group_df in df.groupby(['band', 'mode']):
    # 按evt分组数据
    evt_groups = {evt: group_df[group_df['evt'] == evt]['values'].values
                  for evt in sorted(group_df['evt'].unique())}

    # 对每个时间点进行T检验
    for time_point in times:
        # 提取该时间点每个evt的数据
        time_data = {}
        for evt in EVENTS:  # evt 0-4
            evt_time_data = group_df[(group_df['evt'] == evt) &
                                     (group_df['times'] == time_point)]
            time_data[evt] = evt_time_data['values'].values

        # print(time_data)
        # ANOVA on time_data

        # 对所有evt对进行两两比较
        for evt1, evt2 in combinations(EVENTS, 2):
            if len(time_data[evt1]) > 0 and len(time_data[evt2]) > 0:
                # 进行独立样本T检验
                t_stat, p_value = stats.ttest_ind(
                    time_data[evt1],
                    time_data[evt2],
                    equal_var=False  # Welch's t-test (不假设方差齐性)
                )

                # 计算效应量 (Cohen's d)
                mean1, mean2 = np.mean(
                    time_data[evt1]), np.mean(time_data[evt2])
                std1, std2 = np.std(time_data[evt1], ddof=1), np.std(
                    time_data[evt2], ddof=1)
                n1, n2 = len(time_data[evt1]), len(time_data[evt2])

                # 合并标准差
                pooled_std = np.sqrt(
                    ((n1-1)*std1**2 + (n2-1)*std2**2) / (n1 + n2 - 2))
                cohens_d = (mean1 - mean2) / \
                    pooled_std if pooled_std != 0 else 0

                # 保存结果
                results.append({
                    'band': band,
                    'mode': mode,
                    'time': time_point,
                    'evt1': evt1,
                    'evt2': evt2,
                    't_statistic': t_stat,
                    'p_value': p_value,
                    'cohens_d': cohens_d,
                    'mean1': mean1,
                    'mean2': mean2,
                    'std1': std1,
                    'std2': std2,
                    'n1': n1,
                    'n2': n2
                })

# 转换为DataFrame
results_df = pd.DataFrame(results)
print(results_df)

# %%
indexes = []
for i, row in results_df.iterrows():
    if row['p_value'] > 0.05:
        continue
    key = (row['mode'], row['band'])
    ts = np.array(times)[cluster_timelines[key] == 1]
    if row['time'] in ts:
        indexes.append(i)
filtered_results_df = results_df.loc[indexes].reset_index(drop=True)
print(filtered_results_df, len(filtered_results_df))

# %% ---- 2026-01-20 ------------------------
# Pending
df = filtered_results_df.copy()
df['compare'] = df['evt1'] + '_vs_' + df['evt2']
g = sns.FacetGrid(df, col="band", row="mode", sharey=True,
                  sharex=True, margin_titles=True, height=4)
g.map_dataframe(sns.scatterplot, x="time", y="compare", hue="compare", s=10)
g.add_legend()

# %%

# %%

# %%
