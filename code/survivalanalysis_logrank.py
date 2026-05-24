import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from lifelines import KaplanMeierFitter, CoxPHFitter
from lifelines.statistics import multivariate_logrank_test

OUTPUT_DIR = "survival_results"
os.makedirs(OUTPUT_DIR, exist_ok=True)

def get_solar_activity():
    print("正在从 CelesTrak 获取历史太阳活动数据 (F10.7)...")
    url = "https://celestrak.org/SpaceData/SW-All.csv"
    try:
        sw_df = pd.read_csv(url)
        sw_df['DATE'] = pd.to_datetime(sw_df['DATE'], errors='coerce').dt.tz_localize(None)
        sw_df = sw_df[['DATE', 'F10.7_OBS']].dropna()
        sw_df.set_index('DATE', inplace=True)
        sw_df.sort_index(inplace=True)
        return sw_df
    except Exception as e:
        print(f"太阳活动数据获取失败: {e}")
        return None

def build_survival_dataset(raw_df, sw_df):
    print("正在将时间序列转换为生存分析截面数据...")
    
    sat_df = raw_df.groupby('NORAD_CAT_ID').agg(
        Survival_Days=('SURVIVAL_DURATION_DAYS', 'max'),
        Event=('SURVIVAL_EVENT', 'last'), 
        Launch_Date=('EPOCH', 'min'),
        Last_Date=('EPOCH', 'max'),
        Mean_Inclination=('INCLINATION', 'mean')
    ).reset_index()
    
    sat_df['Shell_Group'] = pd.cut(
        sat_df['Mean_Inclination'], 
        bins=[0, 54, 100], 
        labels=['Main_53_deg', 'Polar_Other']
    )
    sat_df['Batch_Group'] = pd.cut(
        sat_df['NORAD_CAT_ID'], 
        bins=[0, 47000, 53000, 999999], 
        labels=['V1.0_Early', 'V1.5_Mid', 'V2.0_Late']
    )
    
    sat_df['Solar_Activity_Level'] = 'Low' 
    
    if sw_df is not None:
        for idx, row in sat_df.iterrows():
            end_date = row['Last_Date']
            try:
                # 【强化】：使用 nearest 匹配最近的太空天气日期，绝对不会漏切片
                closest_idx = sw_df.index.get_indexer([end_date], method='nearest')[0]
                closest_date = sw_df.index[closest_idx]
                start_date = closest_date - pd.Timedelta(days=30)
                
                final_period_sw = sw_df.loc[start_date:closest_date]
                if not final_period_sw.empty:
                    mean_f107_final = final_period_sw['F10.7_OBS'].mean()
                    # 设定阈值：退役前30天均值 > 150 视为太阳活跃期
                    sat_df.at[idx, 'Solar_Activity_Level'] = 'High' if mean_f107_final > 150 else 'Low'
            except Exception as e:
                pass 
                
    sat_df = sat_df.dropna()
    sat_df.to_csv(os.path.join(OUTPUT_DIR, "survival_dataset.csv"), index=False)
    return sat_df

def plot_km_and_logrank(sat_df, group_col):
    plt.figure(figsize=(10, 6), dpi=150)
    kmf = KaplanMeierFitter()
    
    groups = sat_df[group_col].unique()
    if len(groups) < 2:
        print(f"警告: {group_col} 只有一种分类 {groups}，无法进行 Log-Rank 检验。")
        return
        
    for group in groups:
        mask = sat_df[group_col] == group
        kmf.fit(sat_df[mask]['Survival_Days'], event_observed=sat_df[mask]['Event'], label=str(group))
        kmf.plot_survival_function(linewidth=2)
        
    plt.title(f'Kaplan-Meier Survival Curve by {group_col}', fontweight='bold')
    plt.xlabel('Days in Orbit')
    plt.ylabel('Survival Probability')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, f'KM_Curve_{group_col}.png'))
    
    results = multivariate_logrank_test(sat_df['Survival_Days'], sat_df[group_col], sat_df['Event'])
    print(f"\n--- Log-Rank 检验结果 ({group_col}) ---")
    print(f"P-value: {results.p_value:.4e}")
    if results.p_value < 0.05:
        print(f"结论: 显著！不同 {group_col} 组别间存在寿命差异。")
    else:
        print(f"结论: 不显著。")

def run_cox_model(sat_df):
    print("\n========== Cox 比例风险模型拟合 ==========")
    cox_data = sat_df[['Survival_Days', 'Event', 'Shell_Group', 'Batch_Group', 'Solar_Activity_Level']]
    
    # 防止无聊的单一变量混入导致矩阵崩溃
    for col in ['Shell_Group', 'Batch_Group', 'Solar_Activity_Level']:
        if cox_data[col].nunique() < 2:
            print(f"警告：{col} 只有一个类别，Cox 模型将忽略此变量。")
            cox_data = cox_data.drop(columns=[col])
            
    cox_data = pd.get_dummies(cox_data, drop_first=True)
    cph = CoxPHFitter(penalizer=0.01)
    cph.fit(cox_data, duration_col='Survival_Days', event_col='Event')
    cph.print_summary()
    
    with open(os.path.join(OUTPUT_DIR, 'Cox_Model_Summary.txt'), 'w', encoding='utf-8') as f:
        f.write(cph.summary.to_string())
    
    plt.figure(figsize=(8, 5), dpi=150)
    cph.plot()
    plt.title('Cox Model Hazard Ratios (Log Scale)', fontweight='bold')
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, 'Cox_Hazard_Ratios.png'))

if __name__ == "__main__":
    print("正在加载原始数据...")
    raw_df = pd.read_csv("data.csv", low_memory=False)
    
    raw_df['EPOCH'] = pd.to_datetime(raw_df['EPOCH'], errors='coerce')
    raw_df = raw_df.dropna(subset=['EPOCH'])
    raw_df['EPOCH'] = raw_df['EPOCH'].dt.tz_localize(None)
    
    # 【最强防御】：拦截 Excel 格式毁坏陷阱
    if raw_df['EPOCH'].dt.year.nunique() == 1 and raw_df['EPOCH'].dt.year.iloc[0] == pd.Timestamp.now().year:
        raise ValueError("\n" + "="*60 + "\n[致命错误] 检测到 data.csv 的日期被 Excel 破坏！\n"
                         "现象：所有年份都变成了今年，说明 Excel 截断了日期只保留了时间。\n"
                         "解决方案：请重新运行 data_preprocessing.py 生成全新数据，\n"
                         "生成后 **绝对不要** 用 Excel 打开并保存它，直接运行本程序！\n" + "="*60)
                         
    raw_df = raw_df[raw_df['EPOCH'].dt.year >= 2019]
    
    sw_df = get_solar_activity()
    survival_df = build_survival_dataset(raw_df, sw_df)
    
    if len(survival_df) > 0:
        plot_km_and_logrank(survival_df, group_col='Batch_Group')
        plot_km_and_logrank(survival_df, group_col='Solar_Activity_Level')
        plot_km_and_logrank(survival_df, group_col='Shell_Group')
        run_cox_model(survival_df)
    else:
        print("有效数据为0。")