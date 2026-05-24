import pandas as pd
import numpy as np

def advanced_preprocess_starlink(file_path):
    print("开始读取原始数据...")
    # 【修复1】加入 low_memory=False，消除 DtypeWarning
    df = pd.read_csv(file_path, low_memory=False)
    
    keep_cols = [
        'NORAD_CAT_ID', 'EPOCH', 'SEMIMAJOR_AXIS', 'BSTAR',
        'INCLINATION', 'ECCENTRICITY', 'ARG_OF_PERICENTER', 'RA_OF_ASC_NODE',
        'DECAY_DATE'
    ]
    df = df[[c for c in keep_cols if c in df.columns]].copy()
    
    df['EPOCH'] = pd.to_datetime(df['EPOCH'])
    df = df.sort_values(by=['NORAD_CAT_ID', 'EPOCH']).reset_index(drop=True)
    df = df.drop_duplicates(subset=['NORAD_CAT_ID', 'EPOCH'], keep='last')
    
    print("正在进行物理边界清洗...")
    df = df[(df['SEMIMAJOR_AXIS'] > 6500) & (df['SEMIMAJOR_AXIS'] < 7500)]
    df = df[(df['ECCENTRICITY'] >= 0) & (df['ECCENTRICITY'] < 0.1)]
    
    processed_dfs = []
    dataset_max_date = df['EPOCH'].max() 
    
    print("正在计算差分与扣除大气阻力趋势...")
    for norad_id, group in df.groupby('NORAD_CAT_ID'):
        group = group.copy()
        
        # 【修复2】在任何数据行被过滤之前，先提取静态生存信息
        decay_date = group['DECAY_DATE'].iloc[0] if 'DECAY_DATE' in group.columns else np.nan
        last_epoch = group['EPOCH'].max()
        first_epoch = group['EPOCH'].min()
        
        group['DELTA_T_DAYS'] = group['EPOCH'].diff().dt.total_seconds() / 86400.0
        group['DELTA_A'] = group['SEMIMAJOR_AXIS'].diff()
        
        # 过滤掉时间间隔极短的观测
        group = group[group['DELTA_T_DAYS'] > 0.01].copy()
        
        # 【修复3】安全防线：如果这颗卫星剩下的数据不足以进行窗口平滑（少于5条），则直接跳过
        if len(group) < 5:
            continue
            
        group['DA_DT'] = group['DELTA_A'] / group['DELTA_T_DAYS']
        
        # 滑动窗口扣除阻力
        group['DRAG_BASELINE'] = group['DA_DT'].rolling(window=80, min_periods=5, center=True).quantile(0.5)
        group['DRAG_BASELINE'] = group['DRAG_BASELINE'].bfill().ffill() 
        group['NET_DA_DT'] = group['DA_DT'] - group['DRAG_BASELINE']
        
        # 判定生存状态
        if pd.notna(decay_date) or (dataset_max_date - last_epoch).days > 30:
            group['SURVIVAL_EVENT'] = 1 
        else:
            group['SURVIVAL_EVENT'] = 0 
            
        group['SURVIVAL_DURATION_DAYS'] = (last_epoch - first_epoch).days
            
        processed_dfs.append(group)
        
    final_df = pd.concat(processed_dfs, ignore_index=True)
    final_df = final_df.dropna(subset=['NET_DA_DT'])
    
    print(f"高级预处理完成！准备进行突变检测的数据共 {len(final_df)} 条。")
    return final_df
# 使用示例：
# 假设你的 CSV 文件名为 "starlink_data.csv"
clean_df = advanced_preprocess_starlink("raw_data.csv")
clean_df.to_csv("data.csv", index=False)