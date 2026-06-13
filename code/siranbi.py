import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
from scipy.stats import chi2

# 创建输出文件夹
OUTPUT_DIR = "conflict_impact_results"
os.makedirs(OUTPUT_DIR, exist_ok=True)
print(f"分析结果将保存在: {OUTPUT_DIR}/")

# ==========================================
# 1. 加载数据与基础定义
# ==========================================
print("正在加载已聚类完毕的特征数据...")
df = pd.read_csv('maneuvers_final_merged.csv', parse_dates=['Maneuver_Start', 'Maneuver_End'])

df['Date'] = df['Maneuver_Start'].dt.normalize()
# 非正常模式：不等于'常规站位保持'即为异常
df['Is_Abnormal'] = df['Final_Mode'] != '常规站位保持'

# ==========================================
# 2. 严格定义时间窗口 （前1年 vs 后2年）
# ==========================================
conflict_date = pd.to_datetime('2022-02-24')
start_date = conflict_date - pd.DateOffset(years=1)  # 2021-02-24
end_date = conflict_date + pd.DateOffset(years=2)    # 2024-02-24

print(f"\n设定分析时间窗口: {start_date.date()} 至 {end_date.date()}")

# 过滤数据，仅保留该三年窗口内的机动事件
df_window = df[(df['Date'] >= start_date) & (df['Date'] <= end_date)].copy()

before_df = df_window[df_window['Date'] < conflict_date]
after_df = df_window[df_window['Date'] >= conflict_date]

# 严格按日历天数计算
n_days_before = (conflict_date - start_date).days
n_days_after = (end_date - conflict_date).days

# ==========================================
# 3. 计算基础指标
# ==========================================
def calculate_metrics(data, days):
    if days == 0:
        return 0, 0, 0
    freq = len(data) / days
    intensity = data['Intensity_Max_Rate'].mean() if len(data) > 0 else 0
    abnormal_ratio = data['Is_Abnormal'].sum() / len(data) if len(data) > 0 else 0
    return freq, intensity, abnormal_ratio

met_before = calculate_metrics(before_df, n_days_before)
met_after  = calculate_metrics(after_df,  n_days_after)

# ==========================================
# 4. Poisson 频率似然比检验
# ==========================================
def poisson_likelihood_ratio_test(before_df, after_df, 
                                  n_days_before, n_days_after):
    """
    检验冲突前后日机动频率的 Poisson 均值是否相等。
    H0: λ_before = λ_after = λ
    H1: λ_before ≠ λ_after
    
    返回：似然比统计量 LR, p 值
    """
    total_before = len(before_df)
    total_after  = len(after_df)
    total_all = total_before + total_after
    n_all = n_days_before + n_days_after
    
    # MLE under H0 (pooled lambda)
    lambda_pooled = total_all / n_all
    # MLE under H1
    lambda_before = total_before / n_days_before
    lambda_after  = total_after  / n_days_after
    
    # log-likelihood under H0
    # log L = sum y_i * log(lambda) - n * lambda  (忽略常数项 ln(y_i!))
    ll0 = total_all * np.log(lambda_pooled) - n_all * lambda_pooled
    
    # log-likelihood under H1
    ll1 = (total_before * np.log(lambda_before) - n_days_before * lambda_before +
           total_after  * np.log(lambda_after)  - n_days_after  * lambda_after)
    
    # Likelihood ratio statistic
    LR = -2 * (ll0 - ll1)
    # p-value from chi-square with 1 df
    p_value = 1 - chi2.cdf(LR, df=1)
    
    return LR, p_value, lambda_before, lambda_after, lambda_pooled

LR_stat, p_val, lam_before, lam_after, lam_pooled = poisson_likelihood_ratio_test(
    before_df, after_df, n_days_before, n_days_after
)

# ==========================================
# 5. 打印统计报表
# ==========================================
print("\n" + "="*60)
print(" 俄乌冲突前后星链星座机动行为对比 (控制时间窗口)")
print("="*60)
print(f"统计区间: 冲突前 {n_days_before} 天 (1年) vs 冲突后 {n_days_after} 天 (2年)")
print("-" * 60)
print(f"【日机动频率】 (Poisson MLE)")
print(f"  冲突前 λ_before = {lam_before:.4f} 次/天")
print(f"  冲突后 λ_after  = {lam_after:.4f} 次/天")
print(f"  合并 λ_pooled  = {lam_pooled:.4f} 次/天")
print(f"  变化率: {((lam_after/lam_before)-1)*100:.1f}%")
print("-" * 60)
print(f"【Poisson 均值似然比检验】")
print(f"  原假设 H0: λ_before = λ_after")
print(f"  似然比统计量 Λ = {LR_stat:.4f}")
print(f"  自由度 = 1")
print(f"  p 值 = {p_val:.6f}")
if p_val < 0.05:
    print(f"  ★ 结论: 拒绝 H0。冲突前后机动频率存在显著差异。")
else:
    print(f"  结论: 不能拒绝 H0。冲突前后机动频率无显著差异。")
print("-" * 60)
print(f"【平均机动强度】 (Intensity_Max_Rate)")
print(f"  冲突前: {met_before[1]:.2f} km/day")
print(f"  冲突后: {met_after[1]:.2f} km/day")
print(f"  变化率: {((met_after[1]/met_before[1])-1)*100:.1f}%" if met_before[1] else "N/A")
print("-" * 60)
print(f"【异常机动占比】")
print(f"  冲突前: {met_before[2]*100:.1f}%")
print(f"  冲突后: {met_after[2]*100:.1f}%")
print("="*60 + "\n")

# ==========================================
# 6. 时间序列可视化 (聚焦 3 年窗口)
# ==========================================
print("正在生成可视化图表...")
window = 30

fig, axs = plt.subplots(3, 1, figsize=(12, 10), sharex=True, dpi=150)
plt.style.use('seaborn-v0_8-whitegrid')

all_dates = pd.date_range(start_date, end_date)

# 子图 1: 频率
daily_freq = df_window.groupby('Date').size().reindex(all_dates, fill_value=0)
axs[0].plot(daily_freq.index, daily_freq.rolling(window=window, center=True).mean(),
            color='#3498db', linewidth=2)
axs[0].axvline(conflict_date, color='#e74c3c', linestyle='--', linewidth=2,
               label='Conflict Starts (2022-02-24)')
axs[0].axhline(y=lam_before, color='#2ecc71', linestyle=':', linewidth=1.5,
               label=f'λ_before={lam_before:.3f}')
axs[0].axhline(y=lam_after, color='#e74c3c', linestyle=':', linewidth=1.5,
               label=f'λ_after={lam_after:.3f}')
axs[0].set_title('Indicator 1: Daily Maneuver Frequency (30-Day Rolling Mean)', 
                 fontsize=13, fontweight='bold')
axs[0].set_ylabel('Total Maneuvers / Day')
axs[0].legend(loc='upper left')

# 子图 2: 强度
daily_intensity = df_window.groupby('Date')['Intensity_Max_Rate'].mean().reindex(all_dates, fill_value=0)
axs[1].plot(daily_intensity.index, daily_intensity.rolling(window=window, center=True).mean(),
            color='#e67e22', linewidth=2)
axs[1].axvline(conflict_date, color='#e74c3c', linestyle='--', linewidth=2)
axs[1].set_title('Indicator 2: Average Maneuver Intensity (Max Rate, km/day)', 
                 fontsize=13, fontweight='bold')
axs[1].set_ylabel('Intensity Max Rate')

# 子图 3: 异常模式频率
daily_abnormal = df_window[df_window['Is_Abnormal']].groupby('Date').size().reindex(all_dates, fill_value=0)
axs[2].plot(daily_abnormal.index, daily_abnormal.rolling(window=window, center=True).mean(),
            color='#9b59b6', linewidth=2)
axs[2].axvline(conflict_date, color='#e74c3c', linestyle='--', linewidth=2)
axs[2].set_title('Indicator 3: Abnormal (Outlier) Mode Frequency', 
                 fontsize=13, fontweight='bold')
axs[2].set_ylabel('Abnormal Events / Day')
axs[2].set_xlabel('Date')

plt.xlim([start_date, end_date])
plt.tight_layout()

output_fig = os.path.join(OUTPUT_DIR, 'conflict_impact_3years_window_with_LR.png')
plt.savefig(output_fig)
print(f"完成！图表已保存至 {output_fig}")