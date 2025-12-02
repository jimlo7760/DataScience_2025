import pandas as pd
import matplotlib.pyplot as plt
import json
import os

# 查找 json_raw 目录中的第一个 JSON 文件
def find_data_file():
    base_dir = './json_raw' if os.path.isdir('./json_raw') else '../crops/json_raw'
    if os.path.isdir(base_dir):
        for fname in os.listdir(base_dir):
            if fname.endswith('.json'):
                return os.path.join(base_dir, fname)
    # fallback: 递归查找 ../../data
    search_root = '../../data'
    for root, dirs, files in os.walk(search_root):
        for fname in files:
            if fname.endswith('.json') and ('crop' in fname.lower() or 'corn' in fname.lower() or 'soybean' in fname.lower() or 'wheat' in fname.lower()):
                return os.path.join(root, fname)
    return None

data_file = find_data_file()
if not data_file:
    print("No crop data file found.")
    exit(1)

print(f"Loading data from: {data_file}")

# 读取 JSON 数据
with open(data_file, 'r') as f:
    data = json.load(f)

# 处理嵌套结构：如果有 'data' 字段则取其值
if isinstance(data, dict) and 'data' in data:
    records = data['data']
elif isinstance(data, list):
    records = data
else:
    print("Unexpected data structure.")
    exit(1)

df = pd.DataFrame(records)

# 查找所有数值列（不包括编码类字段）
numeric_cols = []
exclude_patterns = ['code', 'ansi', 'fips', 'zip', 'time', 'load']
for col in df.columns:
    col_lower = col.lower()
    if any(pattern in col_lower for pattern in exclude_patterns):
        continue
    test_numeric = pd.to_numeric(df[col], errors='coerce')
    if test_numeric.notna().sum() > 0:
        numeric_cols.append(col)

# 查找所有分类字段（字符串类型，值不超过20个唯一值）
categorical_cols = []
groupby_cols = ['commodity_desc', 'class_desc', 'util_practice_desc', 'prodn_practice_desc', 'domain_desc']
for col in groupby_cols:
    if col in df.columns and df[col].nunique() <= 20:
        categorical_cols.append(col)

print(f"Numeric fields: {numeric_cols}")
print(f"Categorical fields for grouping: {categorical_cols}")

if 'year' in df.columns and numeric_cols:
    # 创建图表：1个基础年份图 + 分组图
    n_numeric = len(numeric_cols)
    n_categorical = min(len(categorical_cols), 3)  # 最多3个分类维度
    total_plots = n_numeric + n_categorical
    
    fig, axes = plt.subplots(total_plots, 1, figsize=(14, 4.5 * total_plots))
    if total_plots == 1:
        axes = [axes]
    
    plot_idx = 0
    
    # 按年份聚合的数值字段
    for col in numeric_cols:
        df[col] = pd.to_numeric(df[col], errors='coerce')
        yearly = df.groupby('year')[col].mean()
        
        axes[plot_idx].plot(yearly.index, yearly.values, marker='o', linewidth=2, color='steelblue')
        axes[plot_idx].set_title(f'Average {col} Over Years')
        axes[plot_idx].set_xlabel('Year')
        axes[plot_idx].set_ylabel(f'Average {col}')
        axes[plot_idx].grid(True, alpha=0.3)
        plot_idx += 1
    
    # 按分类字段分组的可视化
    for cat_col in categorical_cols[:n_categorical]:
        if cat_col not in df.columns:
            continue
        # 选择第一个数值列进行分组展示
        numeric_col = numeric_cols[0] if numeric_cols else 'Value'
        df[numeric_col] = pd.to_numeric(df[numeric_col], errors='coerce')
        
        grouped = df.groupby([cat_col, 'year'])[numeric_col].mean().unstack(fill_value=0)
        
        for category in grouped.index:
            axes[plot_idx].plot(grouped.columns, grouped.loc[category], marker='o', label=category, alpha=0.7)
        
        axes[plot_idx].set_title(f'{numeric_col} by {cat_col} Over Years')
        axes[plot_idx].set_xlabel('Year')
        axes[plot_idx].set_ylabel(f'Average {numeric_col}')
        axes[plot_idx].legend(loc='best', fontsize=8)
        axes[plot_idx].grid(True, alpha=0.3)
        plot_idx += 1
    
    plt.tight_layout()
    plt.savefig('crop_visualization.png', dpi=100, bbox_inches='tight')
    print(f"Visualization saved as crop_visualization.png with {total_plots} charts")
    plt.show()
else:
    print(f"Available columns: {df.columns.tolist()}")
    print("Data must contain 'year' and at least one numeric column for visualization.")
