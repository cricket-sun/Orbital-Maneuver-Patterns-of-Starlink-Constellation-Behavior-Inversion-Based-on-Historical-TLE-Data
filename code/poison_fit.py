import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os

# 创建输出文件夹
OUTPUT_DIR = "conflict_impact_results"
os.makedirs(OUTPUT_DIR, exist_ok=True)
print(f"分析结果将保存在: {OUTPUT_DIR}/")

# ==========================================
# 1. 加载【已经聚类好】的合并数据
# ==========================================
print("正在加载已聚类完毕的特征数据...")
df = pd.read_csv('maneuvers_final_merged.csv', parse_dates=['Maneuver_Start', 'Maneuver_End'])

# 标准化日期为天
df['Date'] = df['Maneuver_Start'].dt.normalize()

# 基于你的聚类映射结果，定义非正常模式 (对应原本的噪声点 -1)
# 注意：这里需要与你 maneuvers_final_merged.csv 中该列的实际文字严格一致
df['Is_Abnormal'] = df['Final_Mode'] != '常规位保持'

# ==========================================
# 2. 俄乌冲突前后切分与统计
# ==========================================
conflict_date = pd.to_datetime('2022-02-24')

before_df = df[df['Date'] < conflict_date]
after_df = df[df['Date'] >= conflict_date]

n_days_before = (conflict_date - df['Date'].min()).days
n_days_after = (df['Date'].max() - conflict_date).days

def calculate_metrics(data, days):
    if days == 0: return 0, 0, 0
    freq = len(data) / days
    
    # 【修改点】：直接使用 Intensity_Max_Rate 作为强度指标
    intensity = data['Intensity_Max_Rate'].mean() if len(data) > 0 else 0
    
    abnormal_ratio = data['Is_Abnormal'].sum() / len(data) if len(data) > 0 else 0
    return freq, intensity, abnormal_ratio

met_before = calculate_metrics(before_df, n_days_before)
met_after = calculate_metrics(after_df, n_days_after)

# 打印统计报表
print("\n" + "="*45)
print(" 俄乌冲突前后星链星座机动行为对比 (基于 Max Rate)")
print("="*45)
print(f"统计基准: 冲突前 {n_days_before} 天  vs  冲突后 {n_days_after} 天")
print("-" * 45)
print(f"【日机动频率】 (总机动/天)")
print(f"  冲突前: {met_before[0]:.4f} 次/天")
print(f"  冲突后: {met_after[0]:.4f} 次/天")
print(f"  变化率: {((met_after[0]/met_before[0])-1)*100:.1f}%" if met_before[0] else "N/A")
print("-" * 45)
print(f"【平均机动强度】 (最大半长轴变化率 Intensity_Max_Rate)")
print(f"  冲突前: {met_before[1]:.2f} km/day")
print(f"  冲突后: {met_after[1]:.2f} km/day")
print(f"  变化率: {((met_after[1]/met_before[1])-1)*100:.1f}%" if met_before[1] else "N/A")
print("-" * 45)
print(f"【异常/离群战术机动占比】")
print(f"  冲突前: {met_before[2]*100:.1f}%")
print(f"  冲突后: {met_after[2]*100:.1f}%")
print("="*45 + "\n")

# ==========================================
# 3. 时间序列可视化 (移动平均平滑)
# ==========================================
print("正在生成可视化图表...")
window = 60 # 使用 60 天移动平均线抹平短期波动

fig, axs = plt.subplots(3, 1, figsize=(12, 12), sharex=True, dpi=150)
plt.style.use('seaborn-v0_8-whitegrid')

# 获取全时间轴以填补没有机动的空白日期
all_dates = pd.date_range(df['Date'].min(), df['Date'].max())

# 子图 1: 频率
daily_freq = df.groupby('Date').size().reindex(all_dates, fill_value=0)
axs[0].plot(daily_freq.index, daily_freq.rolling(window=window, center=True).mean(), color='#3498db', linewidth=2.5)
axs[0].axvline(conflict_date, color='#e74c3c', linestyle='--', linewidth=2, label='Russia-Ukraine Conflict Starts')
axs[0].set_title('Indicator 1: Daily Maneuver Frequency (60-Day Rolling Mean)', fontsize=13, fontweight='bold')
axs[0].set_ylabel('Total Maneuvers / Day')
axs[0].legend(loc='upper left')

# 子图 2: 强度 (Intensity_Max_Rate)
daily_intensity = df.groupby('Date')['Intensity_Max_Rate'].mean().reindex(all_dates, fill_value=0)
axs[1].plot(daily_intensity.index, daily_intensity.rolling(window=window, center=True).mean(), color='#e67e22', linewidth=2.5)
axs[1].axvline(conflict_date, color='#e74c3c', linestyle='--', linewidth=2)
axs[1].set_title('Indicator 2: Average Maneuver Intensity (Max Rate, km/day)', fontsize=13, fontweight='bold')
axs[1].set_ylabel('Intensity Max Rate (km/day)')

# 子图 3: 非正常(异常/离群)模式频率
daily_abnormal = df[df['Is_Abnormal']].groupby('Date').size().reindex(all_dates, fill_value=0)
axs[2].plot(daily_abnormal.index, daily_abnormal.rolling(window=window, center=True).mean(), color='#9b59b6', linewidth=2.5)
axs[2].axvline(conflict_date, color='#e74c3c', linestyle='--', linewidth=2)
axs[2].set_title('Indicator 3: Abnormal (Outlier) Mode Frequency', fontsize=13, fontweight='bold')
axs[2].set_ylabel('Abnormal Events / Day')
axs[2].set_xlabel('Date')

plt.tight_layout()
output_fig = os.path.join(OUTPUT_DIR, 'conflict_impact_intensity_maxrate.png')
plt.savefig(output_fig)
print(f"完成！图表已保存至 {output_fig}")