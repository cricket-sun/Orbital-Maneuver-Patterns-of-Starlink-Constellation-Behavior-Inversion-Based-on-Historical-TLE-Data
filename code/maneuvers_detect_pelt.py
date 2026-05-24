import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import ruptures as rpt
import matplotlib.dates as mdates

# 消除 Pandas 替换操作的 FutureWarning 警告
pd.set_option('future.no_silent_downcasting', True)

def detect_maneuvers_with_pelt(df, penalty_multiplier=1.5, thrust_threshold=0.03):
    """
    使用优化的 PELT 算法检测变轨机动，并提取机动特征
    """
    maneuvers_list = []
    
    norad_ids = df['NORAD_CAT_ID'].unique()
    print(f"开始对 {len(norad_ids)} 颗卫星进行极速 PELT 机动检测...")
    
    for sat_id in norad_ids:
        sat_data = df[df['NORAD_CAT_ID'] == sat_id].copy().reset_index(drop=True)
        
        # 数据清洗与物理截断
        sat_data.replace([np.inf, -np.inf], np.nan, inplace=True)
        sat_data.dropna(subset=['NET_DA_DT', 'SEMIMAJOR_AXIS', 'EPOCH'], inplace=True)
        sat_data['NET_DA_DT'] = sat_data['NET_DA_DT'].clip(lower=-50.0, upper=50.0)
        
        # 强制转换为 C 连续数组，激活底层极速计算
        signal = np.ascontiguousarray(sat_data['NET_DA_DT'].values.astype(np.float64))
        n_samples = len(signal)
        
        if n_samples < 20:
            continue 
            
        try:
            # 动态惩罚系数计算 (BIC 准则)
            sigma_sq = np.var(signal) 
            dynamic_pen = sigma_sq * np.log(n_samples) * penalty_multiplier
            dynamic_pen = max(dynamic_pen, 5.0) # 设定下限防崩

            # 极速版 PELT 算法
            algo = rpt.Pelt(model="l2", min_size=3, jump=2).fit(signal)
            bkps = algo.predict(pen=dynamic_pen)
            
            start_idx = 0
            current_sat_maneuvers = 0
            
            for end_idx in bkps:
                end_idx_safe = min(end_idx, n_samples - 1)
                segment = sat_data.iloc[start_idx : end_idx_safe + 1]
                
                if len(segment) == 0:
                    break
                
                seg_median_rate = segment['NET_DA_DT'].median()
                
                # 判定阈值并提取特征
                if abs(seg_median_rate) > thrust_threshold:
                    start_time = segment['EPOCH'].iloc[0]
                    end_time = segment['EPOCH'].iloc[-1]
                    duration_days = (end_time - start_time).total_seconds() / 86400.0
                    
                    if duration_days == 0:
                        duration_days = segment['DELTA_T_DAYS'].sum()
                    
                    start_a = segment['SEMIMAJOR_AXIS'].iloc[0]
                    end_a = segment['SEMIMAJOR_AXIS'].iloc[-1]
                    delta_a = end_a - start_a
                    
                    # 增加总幅度校验，滤除微小噪声
                    if abs(delta_a) > 0.1: 
                        maneuvers_list.append({
                            'NORAD_CAT_ID': sat_id,
                            'Maneuver_Start': start_time,
                            'Maneuver_End': end_time,
                            'Duration_Days': duration_days,
                            'Start_SemiMajorAxis': start_a,
                            'End_SemiMajorAxis': end_a,
                            'Delta_a': delta_a,
                            'Intensity_Max_Rate': segment['DA_DT'].abs().max(), 
                            'Avg_BSTAR_During': segment['BSTAR'].mean(),
                            'Maneuver_Type': 'Orbit Raise' if delta_a > 0 else 'Lowering/Avoidance'
                        })
                        current_sat_maneuvers += 1
                
                start_idx = end_idx_safe
                
            print(f" -> 卫星 {sat_id} 检测完成，发现 {current_sat_maneuvers} 次机动。")
            
        except Exception as e:
            print(f"卫星 {sat_id} 检测出错: {e}")
            
    maneuvers_df = pd.DataFrame(maneuvers_list)
    print(f"\n全部检测完成！共提取 {len(maneuvers_df)} 条有效机动特征。")
    return maneuvers_df


def plot_maneuvers(df, maneuvers_df, target_sats):
    """
    可视化指定卫星的轨道半长轴演化及机动标注
    """
    if len(target_sats) == 0:
        print("没有指定需要可视化的卫星。")
        return
        
    # 设置高对比度绘图风格
    plt.style.use('seaborn-v0_8-whitegrid')
    fig, axes = plt.subplots(len(target_sats), 1, figsize=(15, 6 * len(target_sats)), dpi=100)
    if len(target_sats) == 1: 
        axes = [axes]
    
    for idx, sat_id in enumerate(target_sats):
        ax = axes[idx]
        
        sat_data = df[df['NORAD_CAT_ID'] == sat_id]
        if sat_data.empty:
            continue
            
        # 绘制基础轨道衰减曲线
        ax.plot(sat_data['EPOCH'], sat_data['SEMIMAJOR_AXIS'], 
                label='Semi-Major Axis (Natural Decay)', color='#2c3e50', linewidth=1.5, alpha=0.8)
        
        sat_maneuvers = maneuvers_df[maneuvers_df['NORAD_CAT_ID'] == sat_id]
        
        # 标注各个机动区间
        for _, row in sat_maneuvers.iterrows():
            color = '#e74c3c' if row['Maneuver_Type'] == 'Orbit Raise' else '#27ae60'
            
            # 绘制机动时间跨度的半透明色块
            ax.axvspan(row['Maneuver_Start'], row['Maneuver_End'], color=color, alpha=0.2)
            
            # 用醒目的散点标出机动结束时的位置
            ax.scatter(row['Maneuver_End'], row['End_SemiMajorAxis'], 
                       color=color, s=50, edgecolor='white', linewidth=1, zorder=5)
            
        ax.set_title(f"Starlink {sat_id} - Autonomous Maneuver Detection (PELT)", fontsize=16, fontweight='bold', pad=15)
        ax.set_ylabel("Semi-Major Axis $a$ (km)", fontsize=13)
        ax.set_xlabel("Date (Epoch)", fontsize=13)
        ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
        ax.tick_params(labelsize=11)
        
        # 图例处理
        import matplotlib.patches as mpatches
        raise_patch = mpatches.Patch(color='#e74c3c', alpha=0.3, label='Orbit Raise (Station Keeping)')
        lower_patch = mpatches.Patch(color='#27ae60', alpha=0.3, label='Orbit Lowering / Avoidance')
        line_patch = plt.Line2D([0], [0], color='#2c3e50', lw=2, label='Orbital Track')
        ax.legend(handles=[line_patch, raise_patch, lower_patch], loc='best', fontsize=12, frameon=True)

    plt.tight_layout()
    plt.show()

# ==========================================
# 主程序执行入口
# ==========================================
if __name__ == "__main__":
    # 1. 读取预处理后的干净数据 
    # (请确保这里的 csv 文件名与你上一步 data_preprocessing.py 输出的一致)
    print("正在加载时间序列数据...")
    raw_df = pd.read_csv("data.csv", parse_dates=['EPOCH'], low_memory=False)
    
    # 2. 运行机动检测算法
    # 可调节 penalty_multiplier: 
    # 觉得虚假变点多 -> 调大 (如 2.5); 觉得漏检了微小机动 -> 调小 (如 1.0)
    result_df = detect_maneuvers_with_pelt(raw_df, penalty_multiplier=1.5, thrust_threshold=0.03)
    
    # 3. 输出保存 CSV
    if not result_df.empty:
        output_csv = "maneuvers_detected_final.csv"
        result_df.to_csv(output_csv, index=False)
        print(f"\n[成功] 所有机动特征结果已保存至: {output_csv}")
        
        # 4. 可视化输出
        # 自动筛选出检测到多次机动的卫星，随机选取2颗来画图展示
        sats_with_maneuvers = result_df['NORAD_CAT_ID'].value_counts().index.tolist()
        if len(sats_with_maneuvers) > 0:
            sample_sats = sats_with_maneuvers[:2] 
            print(f"\n正在渲染卫星 {sample_sats} 的可视化图像，请稍候...")
            plot_maneuvers(raw_df, result_df, sample_sats)
        else:
            print("没有足够的机动数据进行可视化。")
    else:
        print("\n未检测到任何机动事件。建议在函数调用时调小 penalty_multiplier 测试。")