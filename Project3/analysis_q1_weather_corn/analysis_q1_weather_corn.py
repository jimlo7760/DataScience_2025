"""
Research Question 1: How does weather affect corn yield in Iowa?

This script performs:
1. Data merging (weather + corn yield)
2. Exploratory Data Analysis (EDA)
3. Multiple predictive models
4. Model validation and comparison
"""

import json
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures, StandardScaler
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
import warnings
warnings.filterwarnings('ignore')

# Set style
sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (12, 8)

print("="*80)
print("RESEARCH QUESTION 1: How does weather affect corn yield in Iowa?")
print("="*80)

# ============================================================================
# STEP 1: DATA LOADING AND MERGING
# ============================================================================
print("\n[STEP 1] Loading and merging data...")

# Load Iowa weather data (weekly)
with open('../weather/iowa.json', 'r') as f:
    weather_data = json.load(f)
weather_df = pd.DataFrame(weather_data)
weather_df['date'] = pd.to_datetime(weather_df['date'])
weather_df['year'] = weather_df['date'].dt.year

# Load Iowa corn yield data (annual)
with open('../crops/processed/combined_clean.json', 'r') as f:
    crops_data = json.load(f)
corn_records = crops_data['IA_CORN']['data']
corn_df = pd.DataFrame(corn_records)
corn_df['yield'] = pd.to_numeric(corn_df['Value'], errors='coerce')
corn_df = corn_df[['year', 'yield']].dropna()

print(f"Weather data: {len(weather_df)} weekly records from {weather_df['year'].min()} to {weather_df['year'].max()}")
print(f"Corn data: {len(corn_df)} annual records from {corn_df['year'].min()} to {corn_df['year'].max()}")

# Aggregate weather data for GROWING SEASON (April-September)
print("\nAggregating weather for growing season (April-September)...")
weather_df['month'] = weather_df['date'].dt.month
growing_season = weather_df[weather_df['month'].between(4, 9)]

weather_annual = growing_season.groupby('year').agg({
    'tmax': 'mean',
    'tmin': 'mean',
    'precip': 'sum',
    'solar': 'sum'
}).reset_index()

weather_annual.columns = ['year', 'avg_tmax', 'avg_tmin', 'total_precip', 'total_solar']

# Calculate additional features
weather_annual['avg_temp'] = (weather_annual['avg_tmax'] + weather_annual['avg_tmin']) / 2
weather_annual['temp_range'] = weather_annual['avg_tmax'] - weather_annual['avg_tmin']

# Merge weather with corn yield
merged_df = pd.merge(corn_df, weather_annual, on='year', how='inner')
print(f"\nMerged dataset: {len(merged_df)} years of data")
print(f"Years covered: {merged_df['year'].min()} - {merged_df['year'].max()}")

# Save merged data
merged_df.to_csv('analysis_data_q1.csv', index=False)
print("Saved merged data to: analysis_data_q1.csv")

# ============================================================================
# STEP 2: EXPLORATORY DATA ANALYSIS (EDA)
# ============================================================================
print("\n[STEP 2] Exploratory Data Analysis...")

# Summary statistics
print("\n--- Summary Statistics ---")
print(merged_df.describe().round(2))

# Create EDA visualizations
fig = plt.figure(figsize=(16, 12))

# 2.1 Distributions (Histograms)
print("\nCreating distribution plots...")
weather_vars = ['avg_tmax', 'avg_tmin', 'avg_temp', 'total_precip', 'total_solar']
for i, var in enumerate(weather_vars + ['yield']):
    plt.subplot(3, 3, i+1)
    plt.hist(merged_df[var], bins=15, color='steelblue', alpha=0.7, edgecolor='black')
    plt.xlabel(var)
    plt.ylabel('Frequency')
    plt.title(f'Distribution of {var}')
    plt.grid(alpha=0.3)

plt.tight_layout()
plt.savefig('eda_distributions.png', dpi=150, bbox_inches='tight')
print("Saved: eda_distributions.png")
plt.close()

# 2.2 Correlation Analysis
print("\nComputing correlations...")
correlation_vars = ['yield', 'avg_tmax', 'avg_tmin', 'avg_temp', 'total_precip', 'total_solar', 'temp_range']
corr_matrix = merged_df[correlation_vars].corr()

fig, ax = plt.subplots(figsize=(10, 8))
sns.heatmap(corr_matrix, annot=True, fmt='.3f', cmap='coolwarm', center=0,
            square=True, linewidths=1, cbar_kws={"shrink": 0.8})
plt.title('Correlation Matrix: Weather Variables vs Corn Yield', fontsize=14, fontweight='bold')
plt.tight_layout()
plt.savefig('eda_correlation_matrix.png', dpi=150, bbox_inches='tight')
print("Saved: eda_correlation_matrix.png")
print("\nCorrelations with Yield:")
print(corr_matrix['yield'].sort_values(ascending=False))
plt.close()

# 2.3 Scatter Plots
print("\nCreating scatter plots...")
fig, axes = plt.subplots(2, 3, figsize=(16, 10))
axes = axes.flatten()

for i, var in enumerate(weather_vars):
    axes[i].scatter(merged_df[var], merged_df['yield'], alpha=0.6, s=60, color='darkgreen')
    axes[i].set_xlabel(var, fontsize=11)
    axes[i].set_ylabel('Corn Yield (BU/ACRE)', fontsize=11)
    axes[i].set_title(f'Yield vs {var}', fontweight='bold')
    axes[i].grid(alpha=0.3)

    # Add trend line
    z = np.polyfit(merged_df[var], merged_df['yield'], 1)
    p = np.poly1d(z)
    axes[i].plot(merged_df[var], p(merged_df[var]), "r--", alpha=0.8, linewidth=2)

# Time series plot
axes[5].plot(merged_df['year'], merged_df['yield'], marker='o', linewidth=2, markersize=6, color='navy')
axes[5].set_xlabel('Year', fontsize=11)
axes[5].set_ylabel('Corn Yield (BU/ACRE)', fontsize=11)
axes[5].set_title('Corn Yield Over Time', fontweight='bold')
axes[5].grid(alpha=0.3)

plt.tight_layout()
plt.savefig('eda_scatter_plots.png', dpi=150, bbox_inches='tight')
print("Saved: eda_scatter_plots.png")
plt.close()

# 2.4 Box Plots
fig, axes = plt.subplots(2, 3, figsize=(16, 10))
axes = axes.flatten()

for i, var in enumerate(weather_vars + ['yield']):
    bp = axes[i].boxplot(merged_df[var], patch_artist=True, widths=0.6)
    for patch in bp['boxes']:
        patch.set_facecolor('lightblue')
        patch.set_alpha(0.7)
    axes[i].set_ylabel(var, fontsize=11)
    axes[i].set_title(f'Box Plot: {var}', fontweight='bold')
    axes[i].grid(alpha=0.3, axis='y')

    # Add mean marker
    mean_val = merged_df[var].mean()
    axes[i].plot(1, mean_val, 'r*', markersize=15, label='Mean')
    axes[i].legend()

plt.tight_layout()
plt.savefig('eda_boxplots.png', dpi=150, bbox_inches='tight')
print("Saved: eda_boxplots.png")
plt.close()

# ============================================================================
# STEP 3: PREDICTIVE MODELING
# ============================================================================
print("\n[STEP 3] Building Predictive Models...")

# Prepare features and target
feature_cols = ['avg_tmax', 'avg_tmin', 'total_precip', 'total_solar', 'temp_range']
X = merged_df[feature_cols].values
y = merged_df['yield'].values

# Split data (80% train, 20% test)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
print(f"\nTrain set: {len(X_train)} samples")
print(f"Test set: {len(X_test)} samples")

# Standardize features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Store results
results = {}

# ============================================================================
# MODEL 1: Multiple Linear Regression
# ============================================================================
print("\n--- MODEL 1: Multiple Linear Regression ---")
model1 = LinearRegression()
model1.fit(X_train_scaled, y_train)

y_pred_train_m1 = model1.predict(X_train_scaled)
y_pred_test_m1 = model1.predict(X_test_scaled)

r2_train_m1 = r2_score(y_train, y_pred_train_m1)
r2_test_m1 = r2_score(y_test, y_pred_test_m1)
rmse_m1 = np.sqrt(mean_squared_error(y_test, y_pred_test_m1))
mae_m1 = mean_absolute_error(y_test, y_pred_test_m1)

# Cross-validation
cv_scores_m1 = cross_val_score(model1, X_train_scaled, y_train, cv=5, scoring='r2')

print(f"R² (Training): {r2_train_m1:.4f}")
print(f"R² (Testing): {r2_test_m1:.4f}")
print(f"RMSE: {rmse_m1:.4f} BU/ACRE")
print(f"MAE: {mae_m1:.4f} BU/ACRE")
print(f"Cross-validation R² (5-fold): {cv_scores_m1.mean():.4f} (+/- {cv_scores_m1.std():.4f})")

print("\nFeature Coefficients:")
for feature, coef in zip(feature_cols, model1.coef_):
    print(f"  {feature}: {coef:.4f}")
print(f"  Intercept: {model1.intercept_:.4f}")

results['Linear Regression'] = {
    'model': model1,
    'r2_train': r2_train_m1,
    'r2_test': r2_test_m1,
    'rmse': rmse_m1,
    'mae': mae_m1,
    'cv_scores': cv_scores_m1,
    'predictions': y_pred_test_m1
}

# ============================================================================
# MODEL 2: Polynomial Regression (degree 2)
# ============================================================================
print("\n--- MODEL 2: Polynomial Regression (degree 2) ---")
poly = PolynomialFeatures(degree=2, include_bias=False)
X_train_poly = poly.fit_transform(X_train_scaled)
X_test_poly = poly.transform(X_test_scaled)

model2 = LinearRegression()
model2.fit(X_train_poly, y_train)

y_pred_train_m2 = model2.predict(X_train_poly)
y_pred_test_m2 = model2.predict(X_test_poly)

r2_train_m2 = r2_score(y_train, y_pred_train_m2)
r2_test_m2 = r2_score(y_test, y_pred_test_m2)
rmse_m2 = np.sqrt(mean_squared_error(y_test, y_pred_test_m2))
mae_m2 = mean_absolute_error(y_test, y_pred_test_m2)

print(f"R² (Training): {r2_train_m2:.4f}")
print(f"R² (Testing): {r2_test_m2:.4f}")
print(f"RMSE: {rmse_m2:.4f} BU/ACRE")
print(f"MAE: {mae_m2:.4f} BU/ACRE")

results['Polynomial Regression'] = {
    'model': model2,
    'r2_train': r2_train_m2,
    'r2_test': r2_test_m2,
    'rmse': rmse_m2,
    'mae': mae_m2,
    'predictions': y_pred_test_m2
}

# ============================================================================
# MODEL 3: Random Forest Regression
# ============================================================================
print("\n--- MODEL 3: Random Forest Regression ---")
model3 = RandomForestRegressor(n_estimators=100, max_depth=8, random_state=42)
model3.fit(X_train, y_train)  # RF doesn't need scaling

y_pred_train_m3 = model3.predict(X_train)
y_pred_test_m3 = model3.predict(X_test)

r2_train_m3 = r2_score(y_train, y_pred_train_m3)
r2_test_m3 = r2_score(y_test, y_pred_test_m3)
rmse_m3 = np.sqrt(mean_squared_error(y_test, y_pred_test_m3))
mae_m3 = mean_absolute_error(y_test, y_pred_test_m3)

cv_scores_m3 = cross_val_score(model3, X_train, y_train, cv=5, scoring='r2')

print(f"R² (Training): {r2_train_m3:.4f}")
print(f"R² (Testing): {r2_test_m3:.4f}")
print(f"RMSE: {rmse_m3:.4f} BU/ACRE")
print(f"MAE: {mae_m3:.4f} BU/ACRE")
print(f"Cross-validation R² (5-fold): {cv_scores_m3.mean():.4f} (+/- {cv_scores_m3.std():.4f})")

print("\nFeature Importances:")
for feature, importance in zip(feature_cols, model3.feature_importances_):
    print(f"  {feature}: {importance:.4f}")

results['Random Forest'] = {
    'model': model3,
    'r2_train': r2_train_m3,
    'r2_test': r2_test_m3,
    'rmse': rmse_m3,
    'mae': mae_m3,
    'cv_scores': cv_scores_m3,
    'predictions': y_pred_test_m3
}

# ============================================================================
# STEP 4: MODEL COMPARISON
# ============================================================================
print("\n[STEP 4] Model Comparison...")

comparison_df = pd.DataFrame({
    'Model': ['Linear Regression', 'Polynomial Regression', 'Random Forest'],
    'R² (Train)': [r2_train_m1, r2_train_m2, r2_train_m3],
    'R² (Test)': [r2_test_m1, r2_test_m2, r2_test_m3],
    'RMSE': [rmse_m1, rmse_m2, rmse_m3],
    'MAE': [mae_m1, mae_m2, mae_m3]
})

print("\n" + "="*80)
print("MODEL PERFORMANCE COMPARISON")
print("="*80)
print(comparison_df.to_string(index=False))
print("="*80)

# Save comparison
comparison_df.to_csv('model_comparison_q1.csv', index=False)
print("\nSaved model comparison to: model_comparison_q1.csv")

# ============================================================================
# STEP 5: VISUALIZATIONS
# ============================================================================
print("\n[STEP 5] Creating model visualizations...")

# Predictions vs Actual
fig, axes = plt.subplots(1, 3, figsize=(18, 5))

models_to_plot = [
    ('Linear Regression', y_pred_test_m1, r2_test_m1, rmse_m1),
    ('Polynomial Regression', y_pred_test_m2, r2_test_m2, rmse_m2),
    ('Random Forest', y_pred_test_m3, r2_test_m3, rmse_m3)
]

for i, (name, y_pred, r2, rmse) in enumerate(models_to_plot):
    axes[i].scatter(y_test, y_pred, alpha=0.7, s=100, color='darkblue', edgecolors='black')
    axes[i].plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()],
                 'r--', linewidth=2, label='Perfect Prediction')
    axes[i].set_xlabel('Actual Yield (BU/ACRE)', fontsize=12)
    axes[i].set_ylabel('Predicted Yield (BU/ACRE)', fontsize=12)
    axes[i].set_title(f'{name}\nR² = {r2:.4f}, RMSE = {rmse:.2f}', fontweight='bold')
    axes[i].grid(alpha=0.3)
    axes[i].legend()

plt.tight_layout()
plt.savefig('model_predictions_comparison.png', dpi=150, bbox_inches='tight')
print("Saved: model_predictions_comparison.png")
plt.close()

# Model performance bar chart
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# R² comparison
axes[0].bar(comparison_df['Model'], comparison_df['R² (Test)'], color=['#1f77b4', '#ff7f0e', '#2ca02c'], alpha=0.7)
axes[0].set_ylabel('R² Score', fontsize=12)
axes[0].set_title('Model R² Comparison (Test Set)', fontweight='bold')
axes[0].set_ylim([0, 1])
axes[0].grid(axis='y', alpha=0.3)

# RMSE comparison
axes[1].bar(comparison_df['Model'], comparison_df['RMSE'], color=['#1f77b4', '#ff7f0e', '#2ca02c'], alpha=0.7)
axes[1].set_ylabel('RMSE (BU/ACRE)', fontsize=12)
axes[1].set_title('Model RMSE Comparison (Test Set)', fontweight='bold')
axes[1].grid(axis='y', alpha=0.3)

plt.tight_layout()
plt.savefig('model_performance_bars.png', dpi=150, bbox_inches='tight')
print("Saved: model_performance_bars.png")
plt.close()

# Feature importance (Random Forest)
fig, ax = plt.subplots(figsize=(10, 6))
importances = model3.feature_importances_
indices = np.argsort(importances)[::-1]

ax.bar(range(len(importances)), importances[indices], color='forestgreen', alpha=0.7)
ax.set_xticks(range(len(importances)))
ax.set_xticklabels([feature_cols[i] for i in indices], rotation=45, ha='right')
ax.set_ylabel('Feature Importance', fontsize=12)
ax.set_title('Random Forest: Feature Importance for Corn Yield Prediction', fontweight='bold')
ax.grid(axis='y', alpha=0.3)

plt.tight_layout()
plt.savefig('feature_importance_rf.png', dpi=150, bbox_inches='tight')
print("Saved: feature_importance_rf.png")
plt.close()

print("\n" + "="*80)
print("ANALYSIS COMPLETE!")
print("="*80)
print("\nGenerated files:")
print("  - analysis_data_q1.csv (merged dataset)")
print("  - model_comparison_q1.csv (performance metrics)")
print("  - eda_distributions.png")
print("  - eda_correlation_matrix.png")
print("  - eda_scatter_plots.png")
print("  - eda_boxplots.png")
print("  - model_predictions_comparison.png")
print("  - model_performance_bars.png")
print("  - feature_importance_rf.png")
print("\nNext step: Use these results and visualizations in your written report!")
