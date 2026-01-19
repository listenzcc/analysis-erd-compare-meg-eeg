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

# %%
DATA_DIR = Path('./data/erd.exampleChannels.detail')
MODES = ['meg', 'eeg']
CHANNELS = ['0-C3', '0-MLC42']
EVENTS = ['0', '1', '2', '3', '4']

OUTPUT_DIR = Path('img')
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# %% ---- 2026-01-18 ------------------------
# Function and class


# %% ---- 2026-01-18 ------------------------
# Play ground
# mat0_meg = joblib.load(DATA_DIR / f'meg-4-0-MLC42.dump')['mat']
# mat0_eeg = joblib.load(DATA_DIR / f'eeg-4-0-C3.dump')['mat']

dfs = []
for evt, mode, channel in tqdm(product(EVENTS, MODES, CHANNELS)):
    fpath = DATA_DIR / f'{mode}-{evt}-{channel}.dump'
    if not fpath.is_file():
        continue
    obj = joblib.load(fpath)
    mat = obj['mat']
    freqs = obj['freqs']
    times = obj['times']

    # if mode == 'meg':
    #     mat -= mat0_meg
    # elif mode == 'eeg':
    #     mat -= mat0_eeg
    # else:
    #     raise ValueError(f'Incorrect {mode=}')

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

# %%
table = table[['times', 'values', 'mode', 'evt', 'band']]
print(table)

# %%

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

    # 获取所有时间点
    times = sorted(group_df['times'].unique())

    # 对每个时间点进行T检验
    for time_point in times:
        # 提取该时间点每个evt的数据
        time_data = {}
        for evt in EVENTS:  # evt 0-4
            evt_time_data = group_df[(group_df['evt'] == evt) &
                                     (group_df['times'] == time_point)]
            time_data[evt] = evt_time_data['values'].values

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
significant_df = results_df[results_df['p_value'] < 0.01].copy()
significant_df = significant_df.query(f'time>0 & time<4').copy()
significant_df = significant_df[significant_df['band'].isin(
    ['alpha', 'beta'])].copy()
significant_df

# %%
# 方法1：使用seaborn的relplot（推荐，更简洁）
fig = plt.figure(figsize=(14, 12), dpi=600)

# 创建组合标签
significant_df['comparison'] = significant_df['evt1'].astype(
    str) + ' vs ' + significant_df['evt2'].astype(str)

# 使用relplot创建网格图
g = sns.relplot(
    data=significant_df,
    x='time',
    y='comparison',
    col='band',
    row='mode',
    hue='comparison',
    kind='scatter',
    height=4,
    aspect=1.2,
    facet_kws={'sharex': True, 'sharey': True}
)

# 设置标题和标签
# g.fig.suptitle('Significant Event Comparisons (p < 0.01)',
#                fontsize=14, fontweight='bold')
g.set_axis_labels('Time', 'Comparison')
g.set_titles(col_template='Band: {col_name}', row_template='Mode: {row_name}')

# plt.tight_layout()
plt.savefig(OUTPUT_DIR / 'Significant Event Comparisons.png')
plt.show()


# %%

# 查看结果概览
print(f"总共有 {len(results_df)} 个比较")
print("\n按band和mode分组的比较数量:")
print(results_df.groupby(['band', 'mode']).size())

# 查看显著的比较 (p < 0.05)
significant = results_df[results_df['p_value'] < 0.05]
print(f"\n显著的比较数量 (p < 0.05): {len(significant)}")

# 按时间和频带查看最显著的差异
if len(significant) > 0:
    print("\n最显著的10个差异 (按p值排序):")
    print(significant.nsmallest(10, 'p_value')[['band', 'mode', 'time', 'evt1', 'evt2',
                                               't_statistic', 'p_value', 'cohens_d']])

# 可选：保存结果到CSV
# results_df.to_csv('t_test_results.csv', index=False)

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
