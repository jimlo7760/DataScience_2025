import json
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
import matplotlib.pyplot as plt
import seaborn as sns

# 设置绘图风格
sns.set(style="whitegrid")


def load_yield_data(file_path):
    """加载并清洗产量数据"""
    with open(file_path, 'r') as f:
        data = json.load(f)

    if 'data' in data:
        df = pd.DataFrame(data['data'])
    else:
        print(f"Error: 'data' key not found in {file_path}")
        return None

    # 筛选年度最终统计数据 (reference_period_desc == 'YEAR')
    df = df[df['reference_period_desc'] == 'YEAR']

    # 转换产量数值，处理可能的非数字字符
    df['Value'] = pd.to_numeric(df['Value'], errors='coerce')

    # 提取需要的列: 年份和产量
    df = df[['year', 'Value']].rename(columns={'Value': 'Yield'})
    df['year'] = df['year'].astype(int)

    # 按年份聚合，取平均值 (防止同一年有重复条目)
    df = df.groupby('year').mean().reset_index()

    return df


def load_weather_data(file_path, crop_type):
    """加载并根据作物类型聚合气象数据"""
    with open(file_path, 'r') as f:
        data = json.load(f)

    df = pd.DataFrame(data)
    df['date'] = pd.to_datetime(df['date'])
    df['year'] = df['date'].dt.year
    df['month'] = df['date'].dt.month

    # --- 特征工程: 定义生长季 ---
    if crop_type == 'Wheat':
        # 冬小麦 (Winter Wheat):
        # 生长季跨年，从前一年10月播种到当年6月收割。
        # 定义 "CropYear"：如果是10-12月的数据，归入下一年的作物年份。
        df['CropYear'] = df.apply(lambda x: x['year'] + 1 if x['month'] >= 10 else x['year'], axis=1)
        # 筛选生长季月份: 10, 11, 12 (前一年) + 1, 2, 3, 4, 5, 6 (当年)
        season_months = [10, 11, 12, 1, 2, 3, 4, 5, 6]
        df_season = df[df['month'].isin(season_months)]
        group_col = 'CropYear'
    else:
        # 玉米和大豆 (Corn/Soybeans):
        # 春种秋收，生长季为当年的5月-10月。
        df['CropYear'] = df['year']
        season_months = [5, 6, 7, 8, 9, 10]
        df_season = df[df['month'].isin(season_months)]
        group_col = 'CropYear'

    # --- 聚合气象特征 ---
    # tmax: 生长季平均日最高温
    # precip: 生长季累计总降雨量
    # solar: 生长季平均光照
    weather_agg = df_season.groupby(group_col).agg({
        'tmax': 'mean',
        'precip': 'sum',
        'solar': 'mean'
    }).reset_index().rename(columns={group_col: 'year'})

    return weather_agg


def analyze_crop(yield_file, weather_file, crop_name, state_name):
    print(f"\n--- Processing {crop_name} in {state_name} ---")

    # 1. 加载数据
    df_yield = load_yield_data(yield_file)
    df_weather = load_weather_data(weather_file, crop_name)

    if df_yield is None or df_weather is None:
        return None

    # 2. 合并数据 (Inner Join保证年份对应)
    df_merged = pd.merge(df_yield, df_weather, on='year', how='inner')

    if len(df_merged) < 5:
        print(f"Not enough data points ({len(df_merged)}) for analysis. Skipping.")
        return None

    # 3. 训练机器学习模型 (随机森林回归)
    X = df_merged[['tmax', 'precip', 'solar']]
    y = df_merged['Yield']

    rf = RandomForestRegressor(n_estimators=100, random_state=42)
    rf.fit(X, y)

    # 4. 寻找最佳生长条件 (Model-based Optimization / Simulation)
    # 原理: 创建一个虚拟测试集，只改变一个变量(如温度)，保持其他变量为均值，观察模型预测产量的变化。

    # (A) 最佳温度分析
    tmax_range = np.linspace(X['tmax'].min() - 5, X['tmax'].max() + 5, 100)
    # 构建测试集: 变化的温度 + 平均降雨 + 平均光照
    X_temp_test = pd.DataFrame({
        'tmax': tmax_range,
        'precip': X['precip'].mean(),
        'solar': X['solar'].mean()
    })
    y_pred_temp = rf.predict(X_temp_test)
    optimal_tmax = tmax_range[np.argmax(y_pred_temp)]  # 找到预测产量最大值对应的温度

    # (B) 最佳降雨量分析
    precip_range = np.linspace(X['precip'].min() * 0.5, X['precip'].max() * 1.5, 100)
    # 构建测试集: 平均温度 + 变化的降雨 + 平均光照
    X_precip_test = pd.DataFrame({
        'tmax': X['tmax'].mean(),
        'precip': precip_range,
        'solar': X['solar'].mean()
    })
    y_pred_precip = rf.predict(X_precip_test)
    optimal_precip = precip_range[np.argmax(y_pred_precip)]  # 找到预测产量最大值对应的降雨

    # 5. 绘制结果图 (保存到本地)
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    # 子图1: 温度 vs 产量
    axes[0].plot(tmax_range, y_pred_temp, color='#d62728', lw=3)
    axes[0].axvline(optimal_tmax, color='black', linestyle='--', alpha=0.6, label=f'Optimum: {optimal_tmax:.1f} F')
    axes[0].set_title(f'{crop_name} Yield Response to Temperature', fontsize=14)
    axes[0].set_xlabel('Avg Daily Max Temp (F)', fontsize=12)
    axes[0].set_ylabel('Predicted Yield (Bu/Acre)', fontsize=12)
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    # 子图2: 降雨 vs 产量
    axes[1].plot(precip_range, y_pred_precip, color='#1f77b4', lw=3)
    axes[1].axvline(optimal_precip, color='black', linestyle='--', alpha=0.6, label=f'Optimum: {optimal_precip:.1f} in')
    axes[1].set_title(f'{crop_name} Yield Response to Precipitation', fontsize=14)
    axes[1].set_xlabel('Total Growing Season Precipitation (inch)', fontsize=12)
    axes[1].set_ylabel('Predicted Yield (Bu/Acre)', fontsize=12)
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)

    plt.suptitle(f'Crop: {crop_name} | State: {state_name}', fontsize=16)
    plt.tight_layout()

    # 保存图片文件
    filename = f"{crop_name}_{state_name}_Analysis.png"
    plt.savefig(filename)
    print(f"Plot saved as: {filename}")

    return {
        'Crop': crop_name,
        'State': state_name,
        'Optimal_Temp_F': round(optimal_tmax, 2),
        'Optimal_Precip_Inch': round(optimal_precip, 2)
    }


# --- 主程序 ---
if __name__ == "__main__":
    # 定义任务列表: (产量文件, 天气文件, 作物名称, 州名称)
    tasks = [
        ('IL_SOYBEANS_raw.json', 'illinois.json', 'Soybeans', 'Illinois'),
        ('IA_CORN_raw.json', 'iowa.json', 'Corn', 'Iowa'),
        ('KS_WHEAT_raw.json', 'kansas.json', 'Wheat', 'Kansas')
    ]

    results = []
    for yield_f, weather_f, crop, state in tasks:
        res = analyze_crop(yield_f, weather_f, crop, state)
        if res:
            results.append(res)

    # 打印最终表格
    print("\n" + "=" * 50)
    print("   OPTIMAL GROWING CONDITIONS (ML PREDICTION)")
    print("=" * 50)
    results_df = pd.DataFrame(results)
    print(results_df.to_string(index=False))
    print("=" * 50)
