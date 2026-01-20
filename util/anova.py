import warnings
from scipy.spatial.distance import pdist, squareform
from scipy import stats
import pandas as pd
import numpy as np
import scipy.stats as stats
from scipy.stats import f_oneway
import statsmodels.api as sm
from statsmodels.formula.api import ols
import matplotlib.pyplot as plt
import seaborn as sns

# 假设您的数据已加载为DataFrame df
# 如果从文件读取，可以使用：df = pd.read_csv('your_file.csv')

# 按evt分组提取values数据


def analysis_anova(df):
    groups = []
    for evt_value in sorted(df['evt'].unique()):
        group_data = df[df['evt'] == evt_value]['values'].values
        groups.append(group_data)
        # print(
        #     f"evt {evt_value}: n={len(group_data)}, mean={np.mean(group_data):.4f}, std={np.std(group_data):.4f}")

    # 1. 执行单因素ANOVA
    anova_result = f_oneway(*groups)
    # print("\n=== ANOVA 结果 ===")
    # print(f"F-statistic: {anova_result.statistic:.4f}")
    # print(f"p-value: {anova_result.pvalue:.6f}")

    # 2. 使用statsmodels进行更详细的分析
    model = ols('values ~ C(evt)', data=df).fit()
    anova_table = sm.stats.anova_lm(model, typ=2)
    # print("\n=== 详细ANOVA表 ===")
    # print(anova_table)
    return anova_table


def after_anova_cluster_test(df):
    """
    对ANOVA结果进行聚类置换检验

    参数：
    df: 包含ANOVA结果的DataFrame，必须包含'F'和'p'列

    返回：
    cluster_results: 聚类检验结果
    """
    # ==================== 准备数据 ====================

    # 假设您的数据已加载为DataFrame df
    print("数据概览：")
    print(f"数据形状：{df.shape}")
    print(f"时间点数量：{df['time'].nunique()}")
    print(f"模式：{df['mode'].unique().tolist()}")
    print(f"频带：{df['band'].unique().tolist()}")

    # ==================== 1. 数据准备 ====================
    # 将数据重塑为更易处理的格式

    def prepare_cluster_data(df):
        """
        准备聚类检验所需的数据格式
        返回：字典，键为(mode, band)，值为包含F值和p值的DataFrame
        """
        data_dict = {}

        for (mode_val, band_val), group_df in df.groupby(['mode', 'band']):
            # 按时间排序
            group_df = group_df.sort_values('time')
            data_dict[(mode_val, band_val)] = group_df[[
                'time', 'F', 'p']].reset_index(drop=True)

        return data_dict

    data_dict = prepare_cluster_data(df)

    # ==================== 2. 聚类置换检验函数 ====================

    def cluster_permutation_test(p_series, alpha=0.05, n_permutations=100):
        """
        对时间序列p值进行聚类置换检验

        参数：
        p_series: 时间序列的p值数组
        alpha: 显著性水平
        n_permutations: 置换次数

        返回：
        clusters: 显著聚类信息
        cluster_p_values: 每个聚类的p值
        """
        # 1. 识别初始聚类
        significant_mask = p_series < alpha
        n_timepoints = len(p_series)

        if not np.any(significant_mask):
            return [], []

        # 2. 找到连续显著点形成的聚类
        clusters = []
        in_cluster = False
        current_cluster = []

        for i in range(n_timepoints):
            if significant_mask[i]:
                if not in_cluster:
                    in_cluster = True
                    current_cluster = [i]
                else:
                    current_cluster.append(i)
            else:
                if in_cluster:
                    clusters.append(current_cluster)
                    in_cluster = False

        if in_cluster:
            clusters.append(current_cluster)

        # 3. 计算每个聚类的统计量（负log p值的和）
        cluster_stats = []
        for cluster in clusters:
            # 使用负log p值的和作为聚类统计量
            cluster_stat = -np.sum(np.log(p_series[cluster]))
            cluster_stats.append(cluster_stat)

        if not cluster_stats:
            return [], []

        # 4. 置换检验
        max_cluster_stats_perm = np.zeros(n_permutations)

        for perm in range(n_permutations):
            # 随机打乱时间标签（保持时间结构的方法）
            # 方法1：简单打乱p值序列
            perm_p_series = np.random.permutation(p_series)

            # 在置换数据中寻找聚类
            perm_sig_mask = perm_p_series < alpha

            if np.any(perm_sig_mask):
                # 找到置换数据中的聚类
                perm_clusters = []
                in_perm_cluster = False
                current_perm_cluster = []

                for i in range(n_timepoints):
                    if perm_sig_mask[i]:
                        if not in_perm_cluster:
                            in_perm_cluster = True
                            current_perm_cluster = [i]
                        else:
                            current_perm_cluster.append(i)
                    else:
                        if in_perm_cluster:
                            perm_clusters.append(current_perm_cluster)
                            in_perm_cluster = False

                if in_perm_cluster:
                    perm_clusters.append(current_perm_cluster)

                # 计算置换数据中最大聚类统计量
                perm_cluster_stats = []
                for cluster in perm_clusters:
                    perm_stat = -np.sum(np.log(perm_p_series[cluster]))
                    perm_cluster_stats.append(perm_stat)

                if perm_cluster_stats:
                    max_cluster_stats_perm[perm] = max(perm_cluster_stats)

        # 5. 计算聚类p值
        cluster_p_values = []
        for cluster_stat in cluster_stats:
            p_val = np.sum(max_cluster_stats_perm >=
                           cluster_stat) / n_permutations
            cluster_p_values.append(p_val)

        return clusters, cluster_p_values

    # ==================== 3. 对所有(mode, band)组合进行检验 ====================
    print("\n=== 聚类置换检验结果 ===")

    cluster_results = {}

    for (mode_val, band_val), data_df in data_dict.items():
        print(f"\n--- {mode_val.upper()} - {band_val.upper()} ---")

        p_series = data_df['p'].values
        times = data_df['time'].values

        # 执行聚类置换检验
        clusters, cluster_p_values = cluster_permutation_test(
            p_series, alpha=0.05, n_permutations=1000
        )

        # 存储结果
        cluster_results[(mode_val, band_val)] = {
            'clusters': clusters,
            'cluster_p_values': cluster_p_values,
            'times': times,
            'p_values': p_series,
            'F_values': data_df['F'].values
        }

        # 打印结果
        if clusters:
            print(f"找到 {len(clusters)} 个显著聚类：")
            for i, (cluster, p_val) in enumerate(zip(clusters, cluster_p_values)):
                start_time = times[cluster[0]]
                end_time = times[cluster[-1]]
                duration = end_time - start_time
                n_points = len(cluster)

                # 获取聚类内的平均F值
                cluster_F_values = data_df['F'].values[cluster]
                mean_F = np.mean(cluster_F_values)

                significance = "显著" if p_val < 0.05 else "不显著"
                print(f"  聚类 {i+1}:")
                print(f"    时间范围: {start_time:.2f} 到 {end_time:.2f}")
                print(f"    持续时间: {duration:.2f}")
                print(f"    点数: {n_points}")
                print(f"    平均F值: {mean_F:.3f}")
                print(f"    聚类p值: {p_val:.4f} ({significance})")
        else:
            print("未找到显著聚类")

    return cluster_results
