# Research Question 2 Analysis: Weather Impact on Illinois Soybean Yield

## Overview
This analysis addresses the research question: **"How does weather affect soybean yield in Illinois?"**

This is a **regression/predictive modeling** study that examines the relationship between growing season weather variables (temperature, precipitation, solar radiation) and annual soybean yield in Illinois from 2000-2025.

---

## Research Question 2: How does weather affect soybean yield in Illinois?

**Hypothesis**: Growing season weather conditions (May-September) significantly influence annual soybean yield in Illinois, with temperature, precipitation, and solar radiation acting as key predictive factors.

**Why This Question Matters**:
- Soybeans have different climate requirements than corn
- Illinois is a major soybean producer (top 2 in US)
- Allows comparison of crop-specific weather sensitivities
- Supports diversified agricultural risk management

---

## Directory Structure

```
Project3/
├── analysis_q2_weather_soybean.py           # Main analysis script
├── analysis_data_q2.csv                     # Merged weather + yield dataset
├── model_comparison_q2.csv                  # Model performance metrics
├── eda_distributions_q2.png                 # Histograms
├── eda_correlation_matrix_q2.png            # Correlation heatmap
├── eda_scatter_plots_q2.png                 # Scatter plots
├── eda_boxplots_q2.png                      # Box plots
├── model_predictions_comparison_q2.png      # Actual vs predicted
├── model_performance_bars_q2.png            # Performance comparison
├── feature_importance_rf_q2.png             # Feature ranking
└── README_ANALYSIS_Q2.md                    # This file
```

---

## Data Sources

### Input Data
1. **Weather Data**: `weather/illinois.json`
   - Weekly weather records (2000-2025)
   - Variables: Temperature (max/min), Precipitation, Solar Radiation
   - **Aggregated for soybean growing season: May-September** (different from corn!)

2. **Soybean Yield Data**: `crops/processed/combined_clean.json`
   - Annual soybean yield for Illinois (2000-2025)
   - Unit: Bushels per Acre (BU/ACRE)
   - Source: USDA NASS Survey Data

### Key Difference from Q1
- **Crop**: Soybeans (not corn)
- **State**: Illinois (not Iowa)
- **Growing Season**: May-September (not April-September)
- **Different physiology**: Soybeans are more sensitive to moisture stress and have different heat tolerance

---

## Assignment Question Mapping

### **Question 3a: Research Question 2 (5%)**
**Research Question**: "How does weather affect soybean yield in Illinois?"

**Hypothesis**: We hypothesize that Illinois soybean yields are significantly influenced by growing season weather, with precipitation and temperature playing critical roles. Unlike corn, soybeans may show different sensitivities to weather extremes due to their distinct growth cycle and physiological requirements.

**Where to Find**:
- Script header: `analysis_q2_weather_soybean.py` (lines 1-10)
- This README (Overview section)

**For Your Report (Section 3a)**: Write 3-4 sentences explaining:
1. Why comparing soybeans to corn is valuable
2. Expected differences in weather sensitivity
3. How this addresses crop diversification and risk

---

### **Question 3b: Exploratory Data Analysis (5%)**

#### ✅ Histograms
**File**: `eda_distributions_q2.png`
- Shows distribution of 5 weather variables + soybean yield
- **Assignment Use**: "Figure X: Distribution of weather variables and soybean yield in Illinois (2000-2025)"

#### ✅ Box Plots
**File**: `eda_boxplots_q2.png`
- Box plots with outlier detection for all variables
- **Assignment Use**: "Figure X: Box plots revealing outliers and spread in Illinois soybean data"

#### ✅ Scatter Plots
**File**: `eda_scatter_plots_q2.png`
- 5 weather vs. yield plots + time series
- **Assignment Use**: "Figure X: Relationships between weather variables and soybean yield"

#### ✅ Correlation Matrix
**File**: `eda_correlation_matrix_q2.png`
- Heatmap showing correlations
- **Assignment Use**: "Figure X: Correlation matrix for Illinois soybean dataset"
- **Compare to Q1**: Discuss if soybeans show different correlation patterns than corn

#### ✅ Summary Statistics
**File**: Console output + `analysis_data_q2.csv`
- Descriptive statistics for all variables
- **Assignment Use**: Create comparison table in report:

| Variable | Corn (IA) Mean | Soybean (IL) Mean | Unit |
|----------|----------------|-------------------|------|
| Yield | [from Q1] | [from Q2] | BU/ACRE |
| Avg Tmax | [from Q1] | [from Q2] | °F |
| ... | ... | ... | ... |

---

### **Question 4a: Model Development (5%)**

#### ✅ Same Three Models as Q1 (for comparison)

**Model 1: Multiple Linear Regression**
- 5 weather features (same as Q1)
- Standardized features
- **Comparison point**: Are coefficients different for soybeans vs. corn?

**Model 2: Polynomial Regression (degree 2)**
- 20 features (interactions + quadratics)
- Captures non-linear effects
- **Comparison point**: Do soybeans show stronger/weaker non-linearity?

**Model 3: Random Forest Regression**
- 100 trees, max_depth=8
- Feature importance ranking
- **Comparison point**: Are important features the same for both crops?

**Assignment Use (Section 4a)**: Write 1 page discussing:
1. Same model architecture allows fair comparison between crops
2. Expected differences (e.g., "Soybeans may be more sensitive to precipitation than corn")
3. Feature importance comparison (see `feature_importance_rf_q2.png`)

**Key Figure**: `feature_importance_rf_q2.png`
- Caption: "Figure X: Random Forest feature importance for soybean yield prediction"
- **Compare to Q1**: Create side-by-side comparison in report

---

### **Question 4b: Model Application & Validation (2%)**

#### ✅ Performance Metrics Table
**File**: `model_comparison_q2.csv`

**Assignment Use**: Include table in Section 4b:

| Model | R² (Train) | R² (Test) | RMSE | MAE |
|-------|-----------|----------|------|-----|
| Linear Regression | [computed] | [computed] | [computed] | [computed] |
| Polynomial Regression | [computed] | [computed] | [computed] | [computed] |
| Random Forest | [computed] | [computed] | [computed] | [computed] |

**Critical Comparison**: Create a combined table comparing Q1 vs Q2:

| Crop | Best Model | R² (Test) | RMSE |
|------|-----------|----------|------|
| Corn (IA) | [from Q1] | X.XX | X.XX BU/ACRE |
| Soybean (IL) | [from Q2] | X.XX | X.XX BU/ACRE |

**Discussion**: Are soybeans easier or harder to predict than corn?

#### ✅ Predictions Visualization
**File**: `model_predictions_comparison_q2.png`
- 3 subplots (one per model)
- Actual vs. predicted with perfect prediction line
- **Assignment Use**: "Figure X: Soybean yield predictions by three models"

#### ✅ Performance Bars
**File**: `model_performance_bars_q2.png`
- R² and RMSE comparison bars
- **Assignment Use**: "Figure X: Model performance comparison for soybeans"

#### ✅ Validation Methods (Same as Q1)
- Train/test split (80/20)
- 5-fold cross-validation
- Multiple metrics (R², RMSE, MAE)

**Assignment Use**: Write in Section 4b:
- "We applied identical validation procedures to both crops for fair comparison."
- "Soybean models achieved R² = X.XX vs. corn's R² = X.XX, suggesting [soybeans are more/less predictable]..."
- "Cross-validation showed [Model X] was most stable for soybeans (CV R² = X.XX ± X.XX)."

---

## Comparative Analysis (CRITICAL for Report!)

### For Section 4: Add a "Comparison" subsection

**Compare across questions:**

1. **Correlation Patterns**:
   - Which weather variable is most important for corn? For soybeans?
   - Are the correlations stronger/weaker for one crop?

2. **Model Performance**:
   - Which crop is easier to predict (higher R²)?
   - Does the same model type win for both crops?

3. **Feature Importance**:
   - Do both crops respond similarly to weather?
   - Create side-by-side bar chart of feature importance

4. **Predictability**:
   - Why might one crop be more/less predictable?
   - Climate zone differences (Iowa vs. Illinois)
   - Crop physiology differences

**Example Text for Report**:
```
Comparative Analysis of Corn vs. Soybean Weather Sensitivity

Our analysis reveals distinct weather sensitivity patterns between Iowa corn
and Illinois soybeans. Corn yields showed strongest correlation with [variable]
(r=X.XX), while soybean yields were most influenced by [variable] (r=X.XX).

Model performance differed significantly: [crop] achieved higher predictability
(R²=X.XX vs. X.XX), likely due to [Iowa's more uniform climate / soybeans'
simpler growing requirements / etc.]. Feature importance analysis from Random
Forest models (Figures X and Y) indicates that [temperature/precipitation]
plays a more critical role in [crop] production.

These findings have practical implications for agricultural risk management...
```

---

## Key Findings Checklist (To Document After Running)

### From EDA:
- [ ] Strongest weather-yield correlation for soybeans?
- [ ] Compare to corn: Are soybeans more/less correlated with weather?
- [ ] Any outlier years for soybeans?
- [ ] Time trend in soybean yields?

### From Models:
- [ ] Best performing model for soybeans?
- [ ] Compare to corn: Same best model?
- [ ] R² comparison: Soybeans vs. corn
- [ ] RMSE comparison: Are soybeans easier to predict?
- [ ] Feature importance: What matters most for soybeans?
- [ ] Feature importance comparison: Corn vs. soybeans

---

## How to Use in Your Report

### Section 3a: Research Question 2 (add after Q1)
```
Research Question 2: How does weather affect soybean yield in Illinois?

To examine crop-specific weather sensitivities, we extended our analysis to
soybeans in Illinois. Soybeans have distinct physiological requirements and a
different growing season (May-September vs. corn's April-September). We
hypothesize that soybeans will show different weather sensitivity patterns,
particularly regarding moisture stress and heat tolerance...
```

### Section 3b: EDA for Q2 (add after Q1 EDA)
```
Soybean Yield EDA (Illinois, 2000-2025)

[Include all 4 figure types with Q2 suffix]

Compared to Iowa corn, Illinois soybeans showed [similar/different] distribution
patterns (Figure X). Correlation analysis revealed that [variable] was the
strongest predictor (r=X.XX), differing from corn where [variable] dominated.
This suggests soybeans are [more/less] sensitive to [moisture/temperature]...
```

### Section 4a-4b: Model Results for Q2
```
[Follow same structure as Q1, but emphasize comparisons]

Model Performance Comparison: Corn vs. Soybeans

[Table showing both crops side by side]

Random Forest achieved the best performance for both crops, but with different
R² values (corn: X.XX, soybeans: X.XX). This [X% difference] indicates that
[interpretation]. Feature importance rankings (Figures X and Y) reveal that...
```

---

## Generated Output Files

All files have `_q2` suffix to distinguish from Q1:

| File | Purpose | Assignment Section |
|------|---------|-------------------|
| `analysis_data_q2.csv` | Merged dataset | Reference in methodology |
| `model_comparison_q2.csv` | Performance table | Section 4b (Table) |
| `eda_distributions_q2.png` | Histograms | Section 3b (Figure) |
| `eda_correlation_matrix_q2.png` | Correlations | Section 3b (Figure) |
| `eda_scatter_plots_q2.png` | Relationships | Section 3b (Figure) |
| `eda_boxplots_q2.png` | Outlier detection | Section 3b (Figure) |
| `model_predictions_comparison_q2.png` | Prediction plots | Section 4b (Figure) |
| `model_performance_bars_q2.png` | Performance bars | Section 4b (Figure) |
| `feature_importance_rf_q2.png` | Feature ranking | Section 4a (Figure) |

---

## Next Steps

1. ✅ Run `analysis_q2_weather_soybean.py` to generate all outputs
2. ⬜ Compare Q1 and Q2 results side-by-side
3. ⬜ Create combined comparison figures (optional but impressive)
4. ⬜ Write Sections 3 and 4 of report with both questions integrated
5. ⬜ Discuss crop-specific findings in Conclusions (Section 5)

---

## Critical Discussion Points for Report

**Why comparing crops matters:**
- Different climate sensitivities inform diversification strategies
- Regional climate change impacts may affect crops differently
- Helps farmers make planting decisions

**Expected differences:**
- Soybeans: More sensitive to moisture, later in season
- Corn: More sensitive to early-season temperature (pollination)
- Illinois vs. Iowa: Different climate zones

**Model insights:**
- If same model wins for both: Suggests robust approach
- If different models win: Crop-specific complexity
- Feature importance differences: Actionable insights for farmers

---

## Dependencies

Same as Q1:
```bash
pip install pandas numpy matplotlib seaborn scikit-learn
```

---

## Status

- [x] Script created
- [ ] Script executed
- [ ] Results documented in report
- [ ] Compared with Q1 results
- [ ] Conclusions written
