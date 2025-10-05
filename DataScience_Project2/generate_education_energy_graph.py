import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from scipy import stats

# Read the CSV files
# Education data
la_edu = pd.read_csv('LA_education_2014_2023.csv')
nyc_edu = pd.read_csv('NYC_education_2014_2023.csv')
seattle_edu = pd.read_csv('Seattle_education_2014_2023.csv')

# Energy consumption data
la_energy = pd.read_csv('LA_energy_2014_2023.csv')
nyc_energy = pd.read_csv('NYC_energy_2014_2023.csv')
seattle_energy = pd.read_csv('Seattle_energy_2014_2023.csv')


def merge_city_data(edu_df, energy_df, city_name):
    """
    Merge education and energy data for a city.

    Args:
        edu_df: Education DataFrame
        energy_df: Energy DataFrame
        city_name: Name of the city

    Returns:
        Merged DataFrame with Year, Education level, Consumption, and City
    """
    # Merge on Year
    merged = pd.merge(edu_df, energy_df, on='Year', suffixes=('_edu', '_energy'))
    merged['City'] = city_name
    return merged[['Year', 'Education level', 'Consumption', 'City']]


# Merge data for each city
la_data = merge_city_data(la_edu, la_energy, 'Los Angeles')
nyc_data = merge_city_data(nyc_edu, nyc_energy, 'New York City')
seattle_data = merge_city_data(seattle_edu, seattle_energy, 'Seattle')

# Combine all cities
all_data = pd.concat([la_data, nyc_data, seattle_data], ignore_index=True)

# Define colors for each city
colors = {'Los Angeles': '#3b82f6', 'New York City': '#ef4444', 'Seattle': '#10b981'}

# Print correlation statistics
print("=" * 60)
print("CORRELATION ANALYSIS: EDUCATION vs ENERGY CONSUMPTION")
print("=" * 60)
for city_name, city_data in [('Los Angeles', la_data),
                             ('New York City', nyc_data),
                             ('Seattle', seattle_data)]:
    slope, intercept, r_value, p_value, std_err = stats.linregress(
        city_data['Education level'], city_data['Consumption']
    )
    print(f"\n{city_name}:")
    print(f"  R² (correlation): {r_value ** 2:.4f}")
    print(f"  Slope: {slope:.2f} GWh per year of education")
    print(f"  P-value: {p_value:.6f}")
    print(f"  Correlation: {'Strong' if r_value ** 2 > 0.7 else 'Moderate' if r_value ** 2 > 0.4 else 'Weak'}")


def create_city_plot(city_data, city_name, filename):
    """
    Create a clean single-city scatter plot showing education vs energy consumption.

    Args:
        city_data: DataFrame containing city data
        city_name: Name of the city
        filename: Output filename for the plot
    """
    fig, ax = plt.subplots(figsize=(10, 7))

    # Scatter plot
    ax.scatter(city_data['Education level'], city_data['Consumption'],
               alpha=0.8, s=250, color=colors[city_name],
               edgecolors='black', linewidth=2, zorder=3)

    # Add year labels
    for idx, row in city_data.iterrows():
        ax.annotate(str(int(row['Year'])),
                    (row['Education level'], row['Consumption']),
                    xytext=(8, 8), textcoords='offset points',
                    fontsize=11, fontweight='bold', alpha=0.8)

    # Trend line
    slope, intercept, r_value, p_value, std_err = stats.linregress(
        city_data['Education level'], city_data['Consumption']
    )
    x_trend = np.linspace(city_data['Education level'].min(),
                          city_data['Education level'].max(), 100)
    y_trend = slope * x_trend + intercept
    ax.plot(x_trend, y_trend, '--', color='red', linewidth=3,
            label=f'Trend Line (R² = {r_value ** 2:.3f})', zorder=2, alpha=0.7)

    ax.set_xlabel('Average Education Level (Years)', fontsize=14, fontweight='bold')
    ax.set_ylabel('Energy Consumption (GWh)', fontsize=14, fontweight='bold')
    ax.set_title(f'Energy Consumption vs Education Level\n{city_name} (2014-2023)',
                 fontsize=16, fontweight='bold', pad=20)
    ax.legend(fontsize=12, loc='upper left')
    ax.grid(True, alpha=0.3, linestyle='--')

    # Add text box with statistics
    textstr = f'Correlation: {r_value ** 2:.4f}\nSlope: {slope:.2f} GWh/year'
    props = dict(boxstyle='round', facecolor='wheat', alpha=0.5)
    ax.text(0.98, 0.02, textstr, transform=ax.transAxes, fontsize=10,
            verticalalignment='bottom', horizontalalignment='right', bbox=props)

    plt.tight_layout()
    plt.savefig(filename, dpi=300, bbox_inches='tight')
    print(f"Graph saved as '{filename}'")
    plt.show()
    plt.close()


# Create individual plots for each city
print("\nCreating individual city plots...")
create_city_plot(la_data, 'Los Angeles', 'la_energy_education.png')
create_city_plot(nyc_data, 'New York City', 'nyc_energy_education.png')
create_city_plot(seattle_data, 'Seattle', 'seattle_energy_education.png')
print("\nAll individual city plots created successfully!")


"""
@version: Python 3.10
@created by: Jim Lu
@contact: jimlo7760@gmail.com
@time: 10/5/25
"""
