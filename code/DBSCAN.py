import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import RobustScaler
from sklearn.cluster import HDBSCAN

# ==========================================
# 1. 数据加载与非线性特征工程
# ==========================================
print("正在加载机动特征数据...")
df = pd.read_csv('maneuvers_detected_final.csv')

df['Log_Duration'] = np.log1p(df['Duration_Days'])
df['Log_Intensity'] = np.log1p(df['Intensity_Max_Rate'])
features_for_clustering = ['Delta_a', 'Log_Duration', 'Log_Intensity', 'Start_SemiMajorAxis']
X = df[features_for_clustering].copy()

# ==========================================
# 2. 鲁棒标准化与物理权重
# ==========================================
scaler = RobustScaler()
X_scaled = scaler.fit_transform(X)
weights = np.array([2.5, 1.0, 1.0, 0.5])
X_weighted = X_scaled * weights

# ==========================================
# 3. HDBSCAN 微观聚类
# ==========================================
print("正在运行 HDBSCAN 进行微观聚类...")
clusterer = HDBSCAN(min_cluster_size=5, min_samples=3, cluster_selection_epsilon=0.1)
df['Cluster'] = clusterer.fit_predict(X_weighted)
df['Cluster_Probability'] = clusterer.probabilities_

# ==========================================
# 4. 基于物理意义的宏观合并
# ==========================================
print("正在根据物理特征均值，将微观簇合并为宏观模式...")
original_features = ['Delta_a', 'Duration_Days', 'Intensity_Max_Rate', 'Start_SemiMajorAxis']
cluster_means = df.groupby('Cluster')[original_features].mean()

def merge_to_macro_mode(row):
    """
    根据簇的 Δa 均值映射到物理机动类型
    注意：row.name 是 cluster ID（整数），不是 Δa 值
    """
    delta_a_mean = row['Delta_a']
    
    if row.name == -1:
        return "异常/离群机动"
    elif abs(delta_a_mean) < 2.5:
        return "常规站位保持"
    elif delta_a_mean >= 2.5:
        return "轨道抬升/入轨爬升"
    elif -15 < delta_a_mean <= -2.5:
        return "碰撞规避机动"
    else:  # delta_a_mean <= -15
        return "退役/大幅降轨"

# 生成映射字典：cluster ID → 物理模式名
cluster_to_mode_dict = cluster_means.apply(merge_to_macro_mode, axis=1).to_dict()
df['Final_Mode'] = df['Cluster'].map(cluster_to_mode_dict)

print("\n========== 宏观物理模式合并结果 ==========")
print(df['Final_Mode'].value_counts())

# 保存
df.to_csv('maneuvers_final_merged.csv', index=False)
print("\n[成功] 合并后的模式标签数据已保存至: maneuvers_final_merged.csv")

# ==========================================
# 5. 可视化（修正后的 palette_dict）
# ==========================================
plt.figure(figsize=(10, 6), dpi=150)

# 中文字体设置
plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False
plt.rcParams['font.size'] = 12

# ★ 修正：键名必须与 Final_Mode 列中的实际值完全一致 ★
palette_dict = {
    "常规站位保持":         "#2c3e50",   # 深蓝灰
    "轨道抬升/入轨爬升":     "#e74c3c",   # 红色
    "碰撞规避机动":         "#f39c12",   # 橙色（更醒目）
    "退役/大幅降轨":        "#8e44ad",   # 紫色
    "异常/离群机动":        "#95a5a6",   # 灰色
}

# 绘制散点图
sns.scatterplot(
    data=df,
    x='Duration_Days',
    y='Delta_a',
    hue='Final_Mode',
    palette=palette_dict,
    style='Final_Mode',
    size='Cluster_Probability',
    sizes=(10, 60),
    alpha=0.6,
    edgecolor='w',
    linewidth=0.5
)

# 基准线
plt.axhline(y=0, color='gray', linestyle='--', alpha=0.5)

# 图表装饰
plt.title('Starlink Maneuver Modes (HDBSCAN + Physical Merging)', fontweight='bold', fontsize=14)
plt.xlabel('Duration (Days)', fontsize=12)
plt.ylabel('Delta a (km)', fontsize=12)
plt.grid(True, alpha=0.3)

# 图例外置
plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left', borderaxespad=0., fontsize=10)
plt.tight_layout()

plt.savefig('hdbscan_scatter_merged.png')
print("优化后的聚类散点图已保存为 hdbscan_scatter_merged.png")