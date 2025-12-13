import json
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import learning_curve, train_test_split

# 设置绘图风格
sns.set(style="whitegrid")


# --- 数据加载函数 (复用之前的逻辑) ---
def load_data(yield_file, weather_file, crop_type):
    # 1. Load Yield
    with open(yield_file, 'r') as f:
        y_data = json.load(f)
    df_yield = pd.DataFrame(y_data['data'])
    df_yield = df_yield[df_yield['reference_period_desc'] == 'YEAR']
    df_yield['Value'] = pd.to_numeric(df_yield['Value'], errors='coerce')
    df_yield = df_yield[['year', 'Value']].rename(columns={'Value': 'Yield'})
    df_yield['year'] = df_yield['year'].astype(int)
    df_yield = df_yield.groupby('year').mean().reset_index()

    # 2. Load Weather
    with open(weather_file, 'r') as f:
        w_data = json.load(f)
    df_weather = pd.DataFrame(w_data)
    df_weather['date'] = pd.to_datetime(df_weather['date'])
    df_weather['year'] = df_weather['date'].dt.year
    df_weather['month'] = df_weather['date'].dt.month

    if crop_type == 'Wheat':
        df_weather['CropYear'] = df_weather.apply(lambda x: x['year'] + 1 if x['month'] >= 10 else x['year'], axis=1)
        season_months = [10, 11, 12, 1, 2, 3, 4, 5, 6]
    else:
        df_weather['CropYear'] = df_weather['year']
        season_months = [5, 6, 7, 8, 9, 10]

    df_season = df_weather[df_weather['month'].isin(season_months)]
    weather_agg = df_season.groupby('CropYear').agg({
        'tmax': 'mean', 'precip': 'sum', 'solar': 'mean'
    }).reset_index().rename(columns={'CropYear': 'year'})

    # 3. Merge
    df = pd.merge(df_yield, weather_agg, on='year', how='inner')
    return df


# --- 绘图函数 1: 学习曲线 (Learning Curve) ---
def plot_learning_curve_graph(estimator, X, y, title):
    # 使用 RMSE 作为评估标准 (负值转正值显示)
    train_sizes, train_scores, test_scores = learning_curve(
        estimator, X, y, cv=5, n_jobs=-1,
        train_sizes=np.linspace(0.1, 1.0, 5),
        scoring='neg_root_mean_squared_error'
    )

    train_scores_mean = -np.mean(train_scores, axis=1)
    train_scores_std = np.std(train_scores, axis=1)
    test_scores_mean = -np.mean(test_scores, axis=1)
    test_scores_std = np.std(test_scores, axis=1)

    plt.figure(figsize=(8, 6))
    plt.title(title, fontsize=16)
    plt.xlabel("Training examples", fontsize=14)
    plt.ylabel("RMSE (Lower is better)", fontsize=14)
    plt.grid(True, alpha=0.3)

    # 绘制曲线和置信区间
    plt.fill_between(train_sizes, train_scores_mean - train_scores_std,
                     train_scores_mean + train_scores_std, alpha=0.1, color="r")
    plt.fill_between(train_sizes, test_scores_mean - test_scores_std,
                     test_scores_mean + test_scores_std, alpha=0.1, color="g")
    plt.plot(train_sizes, train_scores_mean, 'o-', color="r", label="Training Score")
    plt.plot(train_sizes, test_scores_mean, 'o-', color="g", label="Cross-validation Score")

    plt.legend(loc="best")
    plt.tight_layout()
    plt.savefig('learning_curve.png')
    print("Saved learning_curve.png")


# --- 绘图函数 2: 残差分析图 (Residuals Plot) ---
def plot_residuals_graph(estimator, X, y, title):
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    estimator.fit(X_train, y_train)
    y_pred_train = estimator.predict(X_train)
    y_pred_test = estimator.predict(X_test)

    residuals_train = y_train - y_pred_train
    residuals_test = y_test - y_pred_test

    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    # 左图: Residuals vs Predicted (检查是否随机分布)
    axes[0].scatter(y_pred_train, residuals_train, c='blue', alpha=0.5, label='Training Data')
    axes[0].scatter(y_pred_test, residuals_test, c='green', marker='s', alpha=0.7, label='Test Data')
    axes[0].axhline(y=0, color='red', linestyle='--')
    axes[0].set_xlabel('Predicted Yield', fontsize=12)
    axes[0].set_ylabel('Residuals', fontsize=12)
    axes[0].set_title('Residuals vs Predicted', fontsize=14)
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    # 右图: Residuals Histogram (检查是否正态分布)
    sns.histplot(residuals_train, color='blue', label='Train', kde=True, ax=axes[1], alpha=0.3)
    sns.histplot(residuals_test, color='green', label='Test', kde=True, ax=axes[1], alpha=0.3)
    axes[1].set_title('Distribution of Residuals', fontsize=14)
    axes[1].set_xlabel('Residual Value', fontsize=12)
    axes[1].legend()

    plt.suptitle(title, fontsize=16)
    plt.tight_layout()
    plt.savefig('residuals_analysis.png')
    print("Saved residuals_analysis.png")


# --- 主程序 ---
crop = 'Soybeans'
state = 'Illinois'
# 请确保当前目录下有这两个文件
df = load_data('IL_SOYBEANS_raw.json', 'illinois.json', crop)

if df is not None and len(df) > 10:
    X = df[['tmax', 'precip', 'solar']]
    y = df['Yield']

    rf_model = RandomForestRegressor(n_estimators=100, max_depth=5, random_state=42)

    plot_learning_curve_graph(rf_model, X, y, f"Learning Curve: {state} {crop}")
    plot_residuals_graph(rf_model, X, y, f"Residual Analysis: {state} {crop}")
else:
    print("Not enough data.")
