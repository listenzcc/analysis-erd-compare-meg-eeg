import pandas as pd
import numpy as np
from scipy import stats
from itertools import combinations


def ttest(df):
    # 假设您的数据已经在df中
    # 如果还没有读取数据，请先读取：
    # df = pd.read_csv('your_data.csv')
    events = sorted(df['evt'].unique())

    # 创建存储结果的列表
    results = []

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
            for evt in events:  # evt 0-4
                evt_time_data = group_df[(group_df['evt'] == evt) &
                                         (group_df['times'] == time_point)]
                time_data[evt] = evt_time_data['values'].values

            # 对所有evt对进行两两比较
            for evt1, evt2 in combinations(events, 2):
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
    return results_df
