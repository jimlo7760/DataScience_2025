import pandas as pd
import matplotlib.pyplot as plt
import json
import os

# 查找处理后的合并作物数据
def find_data_file():
    paths = [
        './processed/combined_clean.json',
        '../crops/processed/combined_clean.json',
        '../../crops/processed/combined_clean.json'
    ]
    for path in paths:
        if os.path.isfile(path):
            return path
    return None

data_file = find_data_file()
if not data_file:
    print("No combined_clean.json found.")
    exit(1)

print(f"Loading yield data from: {data_file}")

# 读取数据
with open(data_file, 'r') as f:
    data = json.load(f)

# 数据应为 {crop: {data: [...]}} 格式
crops_data = {}
for crop_name, crop_info in data.items():
    if isinstance(crop_info, dict) and 'data' in crop_info:
        records = crop_info['data']
    else:
        records = []
    crops_data[crop_name] = records

print(f"Found crops: {list(crops_data.keys())}")

# 为每个作物创建 yield 图表
n_crops = len(crops_data)
fig, axes = plt.subplots(n_crops, 1, figsize=(12, 5 * n_crops))
if n_crops == 1:
    axes = [axes]

colors = ['#1f77b4', '#ff7f0e', '#2ca02c']

for idx, (crop_name, records) in enumerate(crops_data.items()):
    if not records:
        print(f"No data for {crop_name}")
        continue
    
    df = pd.DataFrame(records)
    
    # 过滤 yield 数据
    if 'statisticcat_desc' in df.columns:
        yield_df = df[df['statisticcat_desc'] == 'YIELD'].copy()
    else:
        yield_df = df.copy()
    
    if len(yield_df) == 0:
        print(f"No yield data for {crop_name}")
        continue
    
    # 转换 Value 为数值
    yield_df['Value'] = pd.to_numeric(yield_df['Value'], errors='coerce')
    
    # 按年份排序并绘图
    if 'year' in yield_df.columns:
        yearly_yield = yield_df.groupby('year')['Value'].mean().sort_index()
        
        axes[idx].plot(yearly_yield.index, yearly_yield.values, marker='o', linewidth=2.5, 
                       markersize=8, color=colors[idx % len(colors)], label=crop_name)
        axes[idx].fill_between(yearly_yield.index, yearly_yield.values, alpha=0.2, color=colors[idx % len(colors)])
        
        axes[idx].set_title(f'{crop_name} Yield Over Years', fontsize=14, fontweight='bold')
        axes[idx].set_xlabel('Year', fontsize=12)
        axes[idx].set_ylabel('Yield (BU/ACRE)', fontsize=12)
        axes[idx].grid(True, alpha=0.3, linestyle='--')
        axes[idx].legend(fontsize=11)
        
        # 显示最大最小值
        max_year = yearly_yield.idxmax()
        max_val = yearly_yield.max()
        min_year = yearly_yield.idxmin()
        min_val = yearly_yield.min()
        
        axes[idx].annotate(f'Max: {max_val:.1f}', xy=(max_year, max_val), 
                          xytext=(10, 10), textcoords='offset points', fontsize=9,
                          bbox=dict(boxstyle='round,pad=0.3', facecolor='yellow', alpha=0.3))
        axes[idx].annotate(f'Min: {min_val:.1f}', xy=(min_year, min_val), 
                          xytext=(10, -15), textcoords='offset points', fontsize=9,
                          bbox=dict(boxstyle='round,pad=0.3', facecolor='red', alpha=0.3))

plt.tight_layout()
plt.savefig('crop_yield_visualization.png', dpi=100, bbox_inches='tight')
print("Yield visualization saved as crop_yield_visualization.png")
plt.show()
