import pandas as pd
import matplotlib.pyplot as plt
import json
import os

# 查找天气数据文件（.json 格式，不是 metadata）
def find_data_file():
    # 优先查找当前目录的 .json 文件（非 metadata）
    for fname in os.listdir('.'):
        if fname.endswith('.json') and 'metadata' not in fname:
            return fname
    # fallback: 递归查找上级目录
    for root, dirs, files in os.walk('..'):
        for fname in files:
            if fname.endswith('.json') and 'metadata' not in fname and ('iowa' in fname.lower() or 'kansas' in fname.lower() or 'illinois' in fname.lower()):
                return os.path.join(root, fname)
    return None

data_file = find_data_file()
if not data_file:
    print("No weather data file found.")
    exit(1)

print(f"Loading data from: {data_file}")

# 读取 JSON 数据
with open(data_file, 'r') as f:
    data = json.load(f)

# 处理嵌套结构
if isinstance(data, dict) and 'data' in data:
    records = data['data']
elif isinstance(data, list):
    records = data
else:
    print("Unexpected data structure.")
    exit(1)

df = pd.DataFrame(records)

# 找出所有数值列
numeric_cols = []
for col in df.columns:
    col_lower = col.lower()
    # 跳过日期和编码列
    if col_lower in ['date', 'time', 'location', 'station_id', 'station_name']:
        continue
    test_numeric = pd.to_numeric(df[col], errors='coerce')
    if test_numeric.notna().sum() > 0:
        numeric_cols.append(col)

if 'date' in df.columns and numeric_cols:
    df['date'] = pd.to_datetime(df['date'])
    
    # 为每个数值列创建子图
    n_cols = len(numeric_cols)
    fig, axes = plt.subplots(n_cols, 1, figsize=(12, 4 * n_cols))
    if n_cols == 1:
        axes = [axes]
    
    colors = ['steelblue', 'coral', 'green', 'purple', 'orange']
    for idx, col in enumerate(numeric_cols):
        df[col] = pd.to_numeric(df[col], errors='coerce')
        color = colors[idx % len(colors)]
        
        axes[idx].plot(df['date'], df[col], marker='o', label=col, alpha=0.7, color=color)
        axes[idx].set_title(f'{col} Over Time')
        axes[idx].set_xlabel('Date')
        axes[idx].set_ylabel(col)
        axes[idx].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('weather_visualization.png', dpi=100, bbox_inches='tight')
    print(f"Visualization saved as weather_visualization.png with {n_cols} charts")
    print(f"Analyzed fields: {numeric_cols}")
    plt.show()
else:
    print(f"Available columns: {df.columns.tolist()}")
    print("Data must contain 'date' and at least one numeric column for visualization.")
