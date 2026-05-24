import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import DBSCAN
from sklearn.neighbors import NearestNeighbors
from sklearn.decomposition import PCA

# =============================================
# 1. 读取数据
# =============================================
df = pd.read_csv("maneuvers_detected_final.csv")
print(f"共加载 {len(df)} 条机动记录，{df['NORAD_CAT_ID'].nunique()} 颗卫星")

# =============================================
# 2. 特征工程
# =============================================
df['abs_delta_a'] = np.abs(df['Delta_a'])
df['log_delta_a'] = np.log10(df['abs_delta_a'] + 0.001)      # +0.001 防止 log(0)
df['log_duration'] = np.log10(df['Duration_Days'] + 0.01)
df['start_alt_1000km'] = df['Start_SemiMajorAxis'] / 1000.0   # 千公里单位
df['direction_sign'] = np.sign(df['Delta_a'])                  # +1 提升, -1 降低

# 选定聚类特征
features = ['log_delta_a', 'log_duration', 'start_alt_1000km']
X_raw = df[features].values

# 标准化
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X_raw)

# =============================================
# 3. 用 k‑距离图选 eps
# =============================================
min_samples_candidate = 7
k = min_samples_candidate - 1  # = 4

nn = NearestNeighbors(n_neighbors=min_samples_candidate)
nn.fit(X_scaled)
distances, _ = nn.kneighbors(X_scaled)
k_distances = np.sort(distances[:, k])  # 第k近邻距离排序

plt.figure(figsize=(10, 5))
plt.plot(k_distances, linewidth=1.5)
plt.axhline(y=0.35, color='r', linestyle='--', label='Suggested eps ≈ 0.35')
plt.xlabel("Points sorted by distance", fontsize=12)
plt.ylabel(f"Distance to {min_samples_candidate}th nearest neighbor", fontsize=12)
plt.title("k-Distance Graph for DBSCAN eps Selection", fontsize=14, fontweight='bold')
plt.grid(True, alpha=0.3)
plt.legend()
plt.tight_layout()
plt.savefig("k_distance_graph.png", dpi=150)
plt.show()
print("请观察 k‑距离图，找到曲线拐点对应的 Y 轴值作为 eps")