# Research Question 1 Analysis: Weather Impact on Iowa Corn Yield

## Overview
This analysis addresses the research question: **"How does weather affect corn yield in Iowa?"**

This is a **regression/predictive modeling** study that examines the relationship between growing season weather variables (temperature, precipitation, solar radiation) and annual corn yield in Iowa from 2000-2025.

---

## Directory Structure

```
Project3/
├── analysis_q1_weather_corn.py          # Main analysis script
├── analysis_data_q1.csv                 # Merged weather + yield dataset
├── model_comparison_q1.csv              # Model performance metrics table
├── eda_distributions.png                # Histograms for all variables
├── eda_correlation_matrix.png           # Correlation heatmap
├── eda_scatter_plots.png                # Scatter plots: weather vs yield
├── eda_boxplots.png                     # Box plots for all variables
├── model_predictions_comparison.png     # Actual vs predicted plots
├── model_performance_bars.png           # R² and RMSE comparison bars
├── feature_importance_rf.png            # Random Forest feature ranking
└── README_ANALYSIS_Q1.md                # This file
```

---

## Data Sources

### Input Data
1. **Weather Data**: `weather/iowa.json`
   - Weekly weather records (2000-2025)
   - Variables: Temperature (max/min), Precipitation, Solar Radiation
   - Aggregated for growing season (April-September)

2. **Corn Yield Data**: `crops/processed/combined_clean.json`
   - Annual corn yield for Iowa (2000-2025)
   - Unit: Bushels per Acre (BU/ACRE)
   - Source: USDA NASS Survey Data

### Output Data
- **analysis_data_q1.csv**: Merged dataset (26 years × 7 variables)
  - Columns: year, yield, avg_tmax, avg_tmin, total_precip, total_solar, avg_temp, temp_range

---

## Assignment Question Mapping

### **Question 3a: Research Questions/Hypotheses (5%)**
**Assignment Requirement**: Develop and state two particular questions/hypotheses related to the goal of the investigation.

**This Analysis Addresses**:
- ✅ **Research Question 1**: "How does weather affect corn yield in Iowa?"
- ✅ **Hypothesis**: Growing season weather variables (temperature, precipitation, solar radiation) can predict annual corn yield with statistical significance.

**Where to Find**:
- Script header (lines 1-10 in `analysis_q1_weather_corn.py`)
- This README file (Overview section)

**For Your Report**: Write 3-4 sentences explaining:
1. Why this question is important (food security, climate change, agricultural planning)
2. What you expect to find (e.g., "We hypothesize that higher precipitation and moderate temperatures will correlate with higher yields")
3. How the analysis design answers this (data merging → EDA → predictive models → validation)

---

### **Question 3b: Exploratory Data Analysis (5%)**
**Assignment Requirement**: Conduct Exploratory Data Analysis to examine the distribution of variables using suitable figures (histograms, boxplots, scatter plots, etc.). Compute summary statistics.

#### ✅ Histograms
**File**: `eda_distributions.png`
- **Shows**: Distribution of 5 weather variables + yield (6 histograms)
- **Purpose**: Examine normality, skewness, and range of each variable
- **Assignment Use**: Include in Section 3b with caption: "Figure X: Distribution of weather variables and corn yield (2000-2025)"

#### ✅ Box Plots
**File**: `eda_boxplots.png`
- **Shows**: Box plots for all 6 variables with outlier detection
- **Purpose**: Identify outliers, median, quartiles, and spread
- **Assignment Use**: Include in Section 3b with caption: "Figure X: Box plots showing central tendency and outliers for weather and yield variables"

#### ✅ Scatter Plots
**File**: `eda_scatter_plots.png`
- **Shows**: 6 plots - 5 weather variables vs. yield + time series
- **Purpose**: Visualize linear/non-linear relationships between predictors and target
- **Assignment Use**: Include in Section 3b with caption: "Figure X: Bivariate relationships between weather variables and corn yield with trend lines"

#### ✅ Correlation Matrix
**File**: `eda_correlation_matrix.png`
- **Shows**: Heatmap of correlations between all variables
- **Purpose**: Quantify strength and direction of relationships
- **Key Findings**: (will be computed when script runs)
  - Which weather variable has strongest correlation with yield
  - Multicollinearity between predictors (e.g., avg_tmax and avg_temp)
- **Assignment Use**: Include in Section 3b with caption: "Figure X: Correlation matrix revealing relationships between weather variables and corn yield"

#### ✅ Summary Statistics
**File**: Console output + `analysis_data_q1.csv`
- **Shows**: Mean, std, min, max, 25th/50th/75th percentiles for all variables
- **Purpose**: Describe dataset characteristics quantitatively
- **Assignment Use**: Create a table in Section 3b:

| Variable | Mean | Std | Min | Max | Unit |
|----------|------|-----|-----|-----|------|
| Yield | [computed] | ... | ... | ... | BU/ACRE |
| Avg Tmax | [computed] | ... | ... | ... | °F |
| ... | ... | ... | ... | ... | ... |

---

### **Question 4a: Model Development (5%)**
**Assignment Requirement**: Implement minimum two different types of models. Describe types, patterns/trends, visual approaches, variables, and parameter choices.

#### ✅ Model 1: Multiple Linear Regression
**Type**: Parametric regression
**Variables**: 5 features (avg_tmax, avg_tmin, total_precip, total_solar, temp_range)
**Parameters**:
- Standardized features (StandardScaler)
- OLS estimation
**Output**:
- Coefficients (console output)
- Shows which variables have positive/negative effects on yield

#### ✅ Model 2: Polynomial Regression
**Type**: Non-linear regression
**Variables**: Same 5 features + polynomial terms (degree 2)
**Parameters**:
- Degree = 2 (captures quadratic relationships)
- Total features = 20 (original + interactions + squares)
**Purpose**: Capture non-linear effects (e.g., "too much heat reduces yield")

#### ✅ Model 3: Random Forest Regression
**Type**: Ensemble machine learning (tree-based)
**Variables**: Same 5 features (no scaling needed)
**Parameters**:
- n_estimators = 100 trees
- max_depth = 8 (prevents overfitting)
- random_state = 42 (reproducibility)
**Output**:
- **Feature importance** (which variables matter most)
- See: `feature_importance_rf.png`

**Visual Approach**:
- Correlation matrix guided variable selection
- Scatter plots showed non-linear patterns → justified polynomial/RF models

**Assignment Use**: Write 1 page in Section 4a explaining:
1. Why you chose these 3 models
2. What each model can capture (linear vs. non-linear)
3. Parameter justifications
4. Expected patterns based on EDA

**Figure to Include**: `feature_importance_rf.png`
- Caption: "Figure X: Random Forest feature importance ranking for corn yield prediction"

---

### **Question 4b: Model Application & Validation (2%)**
**Assignment Requirement**: Apply models to assess performance. Discuss confidence in results including statistic measures (R², RMSE, etc.). Discuss validation and optimization.

#### ✅ Performance Metrics Table
**File**: `model_comparison_q1.csv`

| Model | R² (Train) | R² (Test) | RMSE | MAE |
|-------|-----------|----------|------|-----|
| Linear Regression | [computed] | [computed] | [computed] | [computed] |
| Polynomial Regression | [computed] | [computed] | [computed] | [computed] |
| Random Forest | [computed] | [computed] | [computed] | [computed] |

**Assignment Use**: Include this table in Section 4b with explanation:
- **R² (Test)**: Proportion of variance explained (0-1, higher is better)
- **RMSE**: Root Mean Squared Error in BU/ACRE (lower is better)
- **MAE**: Mean Absolute Error in BU/ACRE (lower is better)
- Compare train vs. test R² to check for overfitting

#### ✅ Predictions Visualization
**File**: `model_predictions_comparison.png`
- **Shows**: 3 subplots - actual vs. predicted yield for each model
- **Red dashed line**: Perfect prediction (y = x)
- **Purpose**: Visually assess model accuracy and bias
- **Assignment Use**: Include in Section 4b with caption: "Figure X: Model predictions vs. actual corn yield on test set"

#### ✅ Performance Comparison Bars
**File**: `model_performance_bars.png`
- **Shows**: Bar charts comparing R² and RMSE across models
- **Purpose**: Quick visual comparison of which model performs best
- **Assignment Use**: Include in Section 4b with caption: "Figure X: Model performance comparison"

#### ✅ Validation Methods
**Implemented in Script**:
1. **Train/Test Split** (80/20) - Prevents data leakage
2. **Cross-Validation** (5-fold CV) - For Linear Regression and Random Forest
   - Checks model stability across different data subsets
3. **Multiple Metrics** - R², RMSE, MAE provide different perspectives
4. **Standardization** - For linear models (not RF) to ensure fair coefficient comparison

**Assignment Use**: Write in Section 4b:
- "We validated models using an 80/20 train-test split to prevent overfitting."
- "5-fold cross-validation on the training set showed [Model X] had the most stable performance (CV R² = X.XX ± X.XX)."
- "The best model was [Model Y] with test R² = X.XX and RMSE = X.XX BU/ACRE, meaning predictions were within ±X bushels per acre on average."

---

## Key Findings (To Be Computed)

When you run the script, document these findings in your report:

### From EDA:
- [ ] Which weather variable has the strongest correlation with yield?
- [ ] Are there any outlier years? (from box plots)
- [ ] Is there a time trend in yields? (from time series plot)
- [ ] Are weather variables correlated with each other? (multicollinearity check)

### From Models:
- [ ] Which model has the highest R² on test set?
- [ ] What is the RMSE in practical terms? (e.g., "predictions within ±5 bushels")
- [ ] Which weather variables are most important? (from Random Forest)
- [ ] Are there signs of overfitting? (compare train vs. test R²)

---

## How to Use This Analysis in Your Report

### Section 3a: Research Questions (½ page)
```
Research Question 1: How does weather affect corn yield in Iowa?

We hypothesize that growing season weather conditions significantly influence
annual corn yield, with temperature, precipitation, and solar radiation acting
as primary drivers. This question is critical for agricultural planning and
climate adaptation strategies. To answer this, we merged 26 years of Iowa
weather data with USDA corn yield records and implemented predictive models
ranging from linear regression to ensemble methods.
```

### Section 3b: EDA (2-3 pages)
1. **Introduction**: "We conducted comprehensive EDA on [N] years of merged data..."
2. **Include figures**: eda_distributions.png, eda_boxplots.png, eda_scatter_plots.png, eda_correlation_matrix.png
3. **Summary statistics table** (from console output)
4. **Key observations**:
   - "Yield shows [normal/skewed] distribution with mean X and std Y..."
   - "Strong positive correlation (r=X.XX) between [variable] and yield..."
   - "Scatter plots reveal [linear/non-linear] relationships..."

### Section 4a: Model Development (1.5 pages)
1. **Model selection rationale**: "We implemented three models to capture different relationship types..."
2. **Describe each model** with parameters
3. **Include figure**: feature_importance_rf.png
4. **Explain variable choices**: "Based on correlation analysis, we selected 5 weather features..."

### Section 4b: Model Performance (1.5 pages)
1. **Include table**: model_comparison_q1.csv
2. **Include figure**: model_predictions_comparison.png and model_performance_bars.png
3. **Interpret results**: "The [best model] achieved R²=X.XX, explaining XX% of yield variance..."
4. **Discuss validation**: "Cross-validation confirmed model stability..."
5. **Confidence assessment**: "RMSE of X.XX bushels means we can predict yield within ±X bushels..."

---

## Dependencies

To run `analysis_q1_weather_corn.py`:
```bash
pip install pandas numpy matplotlib seaborn scikit-learn
```


---

## Notes

- This analysis covers approximately **15-20%** of the total assignment grade
- You still need a second research question to complete Question 3 and 4
- All visualizations are high-resolution (150 DPI) suitable for report inclusion
- The script includes extensive comments for understanding methodology
- Results are reproducible (random_state=42 set for all models)

---

## Citation

If referencing this analysis in your report:
- Weather Data Source: Open-Meteo Historical Weather API (ERA5 Reanalysis)
- Crop Data Source: USDA National Agricultural Statistics Service (NASS)
- Models: scikit-learn 1.x (cite: Pedregosa et al., 2011)
