# Research Question 3 Analysis: Weather Impact on Kansas Wheat Yield

## Overview
This analysis addresses the **BONUS** research question: **"How does weather affect wheat yield in Kansas?"**

**IMPORTANT**: This is the third research question. The assignment only requires TWO, so this is **extra work** that strengthens your analysis!

---

## Why This Question is Special

### 🌾 Wheat has a UNIQUE Growing Cycle

Unlike corn and soybeans (spring-planted, summer-harvested):
- **Winter wheat** is planted in **fall** (September-October)
- Germinates and establishes before winter
- Goes **dormant** during winter months
- Resumes growth in **spring** (March-May)
- Harvested in **early summer** (June)

**This means the growing season spans TWO calendar years!**

### 🗺️ Kansas Climate is Different
- **More continental**: Hotter summers, colder winters
- **Drier**: Less precipitation than Iowa/Illinois
- **Great Plains**: Prone to drought and temperature extremes
- **Wind**: Higher wind speeds affect evapotranspiration

### 📊 Why This Strengthens Your Analysis
1. **Complete dataset coverage**: All 3 states, all 3 crops
2. **Climate diversity**: Humid (IA/IL) vs. Semi-arid (KS)
3. **Crop diversity**: Different growing seasons and requirements
4. **Methodological rigor**: Shows you can handle complex temporal relationships

---

## Research Question 3: How does weather affect wheat yield in Kansas?

**Hypothesis**: Kansas wheat yields are significantly influenced by weather across the entire growing cycle (fall establishment, winter survival, spring growth). We expect different sensitivity patterns compared to corn and soybeans due to wheat's unique phenology and Kansas's drier climate.

---

## Data Sources

### Input Data
1. **Weather Data**: `weather/kansas.json`
   - Weekly weather records (2000-2025)
   - Variables: Temperature (max/min), Precipitation, Solar Radiation
   - **Aggregated across THREE seasons**:
     - **Fall** (Sept-Nov, previous year): Planting and establishment
     - **Winter** (Dec-Feb): Dormancy and cold hardiness
     - **Spring** (Mar-May): Active growth and grain fill

2. **Wheat Yield Data**: `crops/processed/combined_clean.json`
   - Annual wheat yield for Kansas (2000-2025)
   - Unit: Bushels per Acre (BU/ACRE)
   - Source: USDA NASS Survey Data

### Key Methodological Challenge
For harvest year Y, we use weather from:
- Fall Y-1 (Sept-Nov)
- Winter Y-1/Y (Dec-Feb)
- Spring Y (Mar-May)

The script handles this cross-year aggregation automatically.

---

## Directory Structure

```
Project3/
├── analysis_q3_weather_wheat.py             # Main analysis script
├── analysis_data_q3.csv                     # Merged weather + yield dataset
├── model_comparison_q3.csv                  # Model performance metrics
├── eda_distributions_q3.png                 # Histograms
├── eda_correlation_matrix_q3.png            # Correlation heatmap
├── eda_scatter_plots_q3.png                 # Scatter plots
├── eda_boxplots_q3.png                      # Box plots
├── model_predictions_comparison_q3.png      # Actual vs predicted
├── model_performance_bars_q3.png            # Performance comparison
├── feature_importance_rf_q3.png             # Feature ranking
└── README_ANALYSIS_Q3.md                    # This file
```

---

## Assignment Question Mapping

### **Question 3a: Research Question 3 (BONUS)**
**Research Question**: "How does weather affect wheat yield in Kansas?"

**Hypothesis**: Wheat's multi-season growing cycle makes it sensitive to weather conditions across fall, winter, and spring. We hypothesize that spring precipitation and winter minimum temperatures are critical factors, differing from the summer-focused patterns of corn and soybeans.

**For Your Report (Section 3a)**: Add a third subsection:
```
Research Question 3 (Additional Analysis): How does weather affect wheat yield in Kansas?

To provide comprehensive coverage of our dataset and explore crop-specific
climate sensitivities, we extended our analysis to winter wheat in Kansas.
Wheat presents a unique analytical challenge due to its multi-season growing
cycle, requiring weather aggregation across fall planting, winter dormancy,
and spring growth phases. Kansas's semi-arid climate also contrasts with the
humid conditions of Iowa and Illinois, allowing us to examine how regional
climate differences affect prediction accuracy...
```

---

### **Question 3b: EDA for Wheat (BONUS)**

#### ✅ Histograms
**File**: `eda_distributions_q3.png`
- Includes seasonal variables (spring_tmax, winter_tmin, fall_precip, etc.)
- **Assignment Use**: "Figure X: Distribution of seasonal weather variables and wheat yield in Kansas"

#### ✅ Box Plots
**File**: `eda_boxplots_q3.png`
- **Assignment Use**: "Figure X: Box plots for Kansas wheat dataset showing seasonal weather patterns"

#### ✅ Scatter Plots
**File**: `eda_scatter_plots_q3.png`
- Shows relationships with key seasonal variables
- **Assignment Use**: "Figure X: Wheat yield relationships with seasonal weather variables"

#### ✅ Correlation Matrix
**File**: `eda_correlation_matrix_q3.png`
- Includes seasonal variables (fall, winter, spring)
- **Assignment Use**: "Figure X: Correlation matrix for Kansas wheat across growing seasons"
- **Key comparison**: Which season matters most for wheat?

#### ✅ Summary Statistics
**File**: `analysis_data_q3.csv`
- Includes seasonal breakdowns
- **Assignment Use**: Expand your comparison table:

| Crop | State | Growing Season | Mean Yield | Key Weather Factor |
|------|-------|----------------|------------|-------------------|
| Corn | Iowa | Apr-Sep | [Q1] | [from Q1] |
| Soybeans | Illinois | May-Sep | [Q2] | [from Q2] |
| Wheat | Kansas | Sep-Jun | [Q3] | [from Q3] |

---

### **Question 4a: Model Development for Wheat (BONUS)**

Same three models as Q1 and Q2 for consistency:
1. **Multiple Linear Regression**
2. **Polynomial Regression (degree 2)**
3. **Random Forest Regression**

**Key Difference**: Features capture multi-season patterns (fall/winter/spring aggregates)

**For Your Report (Section 4a)**:
```
Wheat Model Development

For winter wheat, we aggregated weather data across the full growing cycle:
fall planting (Sept-Nov), winter dormancy (Dec-Feb), and spring growth (Mar-May).
This temporal complexity required careful feature engineering to align weather
data with harvest year yields. We applied the same three model architectures
for consistency, allowing direct comparison of predictive accuracy across crops...
```

**Key Figure**: `feature_importance_rf_q3.png`
- Caption: "Figure X: Random Forest feature importance for wheat yield, revealing multi-season dependencies"

---

### **Question 4b: Model Performance for Wheat (BONUS)**

#### ✅ Performance Metrics
**File**: `model_comparison_q3.csv`

**Critical 3-Way Comparison**: Create comprehensive table in report:

| Crop | State | Best Model | R² (Test) | RMSE | Predictability |
|------|-------|-----------|----------|------|----------------|
| Corn | Iowa | [Q1] | X.XX | X.XX | [High/Med/Low] |
| Soybeans | Illinois | [Q2] | X.XX | X.XX | [High/Med/Low] |
| Wheat | Kansas | [Q3] | X.XX | X.XX | [High/Med/Low] |

**Discussion Points**:
- Which crop is most/least predictable from weather?
- Does Kansas's drier climate make prediction harder?
- Does wheat's complex growing cycle affect predictability?

#### ✅ Visualizations
**Files**:
- `model_predictions_comparison_q3.png`
- `model_performance_bars_q3.png`

**Assignment Use**: Include alongside Q1 and Q2 results for visual comparison

---

## Three-Way Comparative Analysis (CRITICAL!)

### For Section 4: Create a "Cross-Crop Comparison" Subsection

This is where your third question really shines!

#### 1. **Climate Zones**
```
Our analysis spans two distinct climate zones:
- Humid Continental (Iowa, Illinois): Higher precipitation, moderate temperatures
- Semi-Arid (Kansas): Lower precipitation, greater temperature extremes

This allowed us to examine how regional climate affects yield predictability...
```

#### 2. **Growing Seasons**
```
Three distinct phenological patterns:
- Corn (Apr-Sep): Warm season, sensitive to summer heat/drought
- Soybeans (May-Sep): Later planting, sensitive to late-season moisture
- Wheat (Sep-Jun): Multi-season, sensitive to winter cold and spring moisture

Model performance differences reflect these biological realities...
```

#### 3. **Feature Importance Comparison**
Create a side-by-side-by-side figure showing Random Forest importances for all three crops:

| Feature | Corn Importance | Soybeans Importance | Wheat Importance |
|---------|----------------|---------------------|------------------|
| Avg Tmax | [Q1] | [Q2] | [Q3] |
| Avg Tmin | [Q1] | [Q2] | [Q3] |
| Total Precip | [Q1] | [Q2] | [Q3] |
| Total Solar | [Q1] | [Q2] | [Q3] |
| Temp Range | [Q1] | [Q2] | [Q3] |

**Key Insight**: "Precipitation ranks [highest/lowest] for wheat, reflecting Kansas's water-limited environment..."

#### 4. **Predictability Patterns**
```
Predictive Model Performance Across Crops

R² values ranged from X.XX (wheat) to X.XX (corn/soybeans), indicating that
[crop] yields are most predictable from weather alone. This may be due to:
1. [Iowa/Illinois] having more consistent weather patterns
2. [Crop]'s simpler growing season reducing confounding factors
3. Kansas's semi-arid climate introducing greater uncertainty

The consistent performance of [Model Type] across all three crops suggests
that [linear/non-linear] relationships dominate weather-yield interactions
regardless of crop type or region...
```

---

## Key Findings Checklist

### From EDA:
- [ ] Which season (fall/winter/spring) most strongly correlates with wheat yield?
- [ ] How does Kansas precipitation compare to Iowa/Illinois?
- [ ] Are Kansas temperatures more extreme (greater temp_range)?
- [ ] Time trend: Is wheat yield improving as fast as corn/soybeans?

### From Models:
- [ ] Best model for wheat?
- [ ] Is wheat MORE or LESS predictable than corn/soybeans?
- [ ] Which seasonal period matters most (from feature importance)?
- [ ] Compare: Do all three crops have same "best model"?

### Cross-Crop Insights:
- [ ] Which crop is easiest to predict? Hardest?
- [ ] Does the same weather variable (e.g., precip) matter equally for all crops?
- [ ] Does Kansas's drier climate reduce R² compared to IA/IL?
- [ ] Do multi-season crops (wheat) have different model requirements?

---

## How to Use in Your Report

### Section 3a: Add Third Research Question
After Q1 (corn) and Q2 (soybeans), add:
```
To comprehensively evaluate weather-yield relationships across multiple crops
and climate zones, we conducted a third analysis examining winter wheat in
Kansas. This addition allows us to compare humid vs. semi-arid climates and
single-season vs. multi-season crops...
```

### Section 3b: Add Wheat EDA
After corn and soybean EDA, add:
```
Kansas Wheat EDA (2000-2025)

[Include all 4 figure types with _q3 suffix]

Wheat showed distinct patterns from corn and soybeans. Correlation analysis
across seasons (Figure X) revealed that [spring/winter/fall] weather was most
critical (r=X.XX), contrasting with the [summer-focused] patterns of corn and
soybeans. Kansas's lower mean precipitation (X inches vs. [IL/IA] X inches)
is evident in the distributions...
```

### Section 4: Three-Crop Model Comparison
```
Comparative Model Analysis: Corn, Soybeans, and Wheat

[Create master comparison table with all 3 crops]
[Create side-by-side feature importance chart]
[Discuss predictability patterns]

Key Findings:
1. [Crop] was most predictable (R²=X.XX), while [crop] was least (R²=X.XX)
2. Precipitation was consistently important but ranked [highest/lowest] for wheat
3. Kansas's semi-arid climate [did/did not] reduce predictive accuracy
4. [Model type] performed best across all crops, suggesting robust methodology

Agricultural Implications:
- [Crop] producers face greatest weather uncertainty
- Kansas wheat farmers should prioritize [spring/winter/fall] management
- Climate change impacts may differ by crop and region...
```

### Section 5: Conclusions
```
This comprehensive study analyzed 26 years of weather-crop data across three
states, three crops, and two climate zones. By examining corn (Iowa), soybeans
(Illinois), and wheat (Kansas), we demonstrated that...

The inclusion of winter wheat revealed unique challenges in modeling multi-season
crops and semi-arid environments. These findings have implications for...
```

---

## Generated Output Files

All files have `_q3` suffix:

| File | Purpose | Report Section |
|------|---------|----------------|
| `analysis_data_q3.csv` | Multi-season merged data | Methodology |
| `model_comparison_q3.csv` | Performance metrics | Section 4b |
| `eda_distributions_q3.png` | Seasonal distributions | Section 3b |
| `eda_correlation_matrix_q3.png` | Seasonal correlations | Section 3b |
| `eda_scatter_plots_q3.png` | Seasonal relationships | Section 3b |
| `eda_boxplots_q3.png` | Outlier detection | Section 3b |
| `model_predictions_comparison_q3.png` | Prediction accuracy | Section 4b |
| `model_performance_bars_q3.png` | Performance comparison | Section 4b |
| `feature_importance_rf_q3.png` | Seasonal importance | Section 4a |

---

## Why This Third Question is Valuable

1. **Demonstrates depth**: Shows you understand crop physiology and climate
2. **Methodological sophistication**: Handling cross-year temporal relationships
3. **Comprehensive coverage**: Uses 100% of your collected data
4. **Real-world relevance**: Multi-crop analysis is how agricultural research is done
5. **Stronger conclusions**: Patterns across 3 crops > patterns from 1-2 crops

