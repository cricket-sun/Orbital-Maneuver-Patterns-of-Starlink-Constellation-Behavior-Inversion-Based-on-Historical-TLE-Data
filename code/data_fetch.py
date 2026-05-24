import requests
import random
import time
import pandas as pd

# ==========================================
# 1. 配置区：请填入你的 Space-Track 账号信息
# ==========================================
USERNAME = ''
PASSWORD = ''

# Space-Track 登录和查询 URL
LOGIN_URL = 'https://www.space-track.org/ajaxauth/login'
BASE_QUERY_URL = 'https://www.space-track.org/basicspacedata/query'

# ==========================================
# 2. 模拟/获取你的全体星链 ID 列表
# ==========================================
# 假设你已经有了一个包含所有星链 NORAD ID 的列表
# 这里我们生成一个模拟列表 (实际操作中，请替换为你爬取到的几千个真实 ID)
# 早期: 44000-47000, 中期: 47000-53000, 近期: 53000-60000
all_starlink_ids = list(range(44235, 47000)) + list(range(47000, 53000)) + list(range(53000, 58000))

# ==========================================
# 3. 分层抽样函数
# ==========================================
def perform_stratified_sampling(id_list, sample_size_per_layer=30):
    """
    将卫星按发射早晚（编号大小）分为三层，每层随机抽取指定数量的样本。
    """
    early_batch = [norad for norad in id_list if norad < 47000]
    mid_batch = [norad for norad in id_list if 47000 <= norad < 53000]
    late_batch = [norad for norad in id_list if norad >= 53000]
    
    # 使用 random.sample 进行无放回随机抽样
    sampled_early = random.sample(early_batch, min(sample_size_per_layer, len(early_batch)))
    sampled_mid = random.sample(mid_batch, min(sample_size_per_layer, len(mid_batch)))
    sampled_late = random.sample(late_batch, min(sample_size_per_layer, len(late_batch)))
    
    print(f"抽样完成：早期抽取 {len(sampled_early)} 颗，中期抽取 {len(sampled_mid)} 颗，近期抽取 {len(sampled_late)} 颗。")
    
    # 合并为一个总的目标下载列表
    return sampled_early + sampled_mid + sampled_late

# ==========================================
# 4. 下载历史数据函数
# ==========================================
def download_historical_tle(target_ids):
    """
    通过 Space-Track API 登录并逐个下载目标卫星的历史 TLE 数据。
    """
    # 使用 Session 保持登录状态
    with requests.Session() as session:
        # 登录 Space-Track
        login_data = {'identity': USERNAME, 'password': PASSWORD}
        response = session.post(LOGIN_URL, data=login_data)
        
        if response.status_code != 200 or 'error' in response.text.lower():
            print("登录失败，请检查账号密码！")
            return
        
        print("Space-Track 登录成功，开始下载数据...")
        
        all_data = [] # 用于存储所有下载到的数据
        
        # 遍历我们抽样出的卫星 ID
        for i, norad_id in enumerate(target_ids):
            print(f"正在下载第 {i+1}/{len(target_ids)} 颗卫星 (NORAD: {norad_id}) 的历史数据...")
            
            # 构建查询 URL：查询这颗卫星 2020-01-01 至今的所有历史 TLE (格式为 CSV)
            # 你可以根据需要修改这里的日期范围，例如 epoch>now-365 表示过去一年
            query = (f"{BASE_QUERY_URL}/class/gp_history/NORAD_CAT_ID/{norad_id}/"
                     f"EPOCH/>2020-01-01/orderby/EPOCH ASC/format/csv")
            
            try:
                res = session.get(query)
                res.raise_for_status() # 检查是否有 HTTP 错误 (如 429 请求过快)
                
                # 如果有数据，我们不保存成纯文本，而是借助 Pandas 方便后续清洗
                if res.text.strip():
                    from io import StringIO
                    df = pd.read_csv(StringIO(res.text))
                    all_data.append(df)
                    print(f" -> 成功获取 {len(df)} 条记录。")
                else:
                    print(" -> 暂无数据。")
                
            except requests.exceptions.RequestException as e:
                print(f" -> 下载出错: {e}")
            
            # 【关键安全机制】：休眠 3 秒，避免触发 429 Too Many Requests 封禁
            time.sleep(3) 

        # 将所有单独的数据框合并成一个总表
        if all_data:
            final_df = pd.concat(all_data, ignore_index=True)
            # 保存为本地 CSV 文件，方便你的突变检测模型直接读取
            final_df.to_csv("sampled_starlink_history.csv", index=False)
            print("\n所有数据下载并合并完成！已保存为 sampled_starlink_history.csv")
        else:
            print("\n未能获取到任何数据。")

# ==========================================
# 5. 主程序执行入口
# ==========================================
if __name__ == "__main__":
    # 设定每层抽取 30 颗，总共 90 颗
    target_sample = perform_stratified_sampling(all_starlink_ids, sample_size_per_layer=30)
    
    # 打印查看抽样结果
    print(f"准备下载的卫星 ID 列表: {target_sample}")
    
    # 启动下载 (记得先把顶部的账号密码改成你自己的)
    download_historical_tle(target_sample)