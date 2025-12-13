# Machine Learning Component: Crop Yield Optimization and Diagnostics

This directory contains the machine learning analysis modules for **Data Science Assignment 3**. The project utilizes Random Forest Regressors to model non-linear interactions between climatic variables and crop yields, specifically focusing on identifying optimal growing thresholds through simulation.

## Prerequisites

To execute the scripts in this module, the following Python libraries are required:

* pandas
* numpy
* scikit-learn
* matplotlib
* seaborn

## File Descriptions

### 1\. ml\_optimization.py (Core Analysis)

**Function:**
This script is responsible for training the Random Forest models for three specific case studies: Iowa Corn, Illinois Soybeans, and Kansas Winter Wheat. It performs a "Model-Based Optimization" by using the trained models to simulate yield responses under varying climatic conditions.

**Outputs:**

  * **Console Output:** Prints the calculated optimal average daily maximum temperature (F) and total seasonal precipitation (inches) for each crop.
  * **Visualizations:** Generates three optimization plots:
      * `Soybeans_Illinois_optimization.png`
      * `Corn_Iowa_optimization.png`
      * `Wheat_Kansas_optimization.png`

### 2\. ml\_diagnostics.py (Model Evaluation)

**Function:**
This script generates diagnostic plots to assess the reliability, variance, and bias of the machine learning models. It specifically addresses model validation requirements.

**Outputs:**

  * **learning\_curve.png:** Displays the relationship between training set size and Root Mean Squared Error (RMSE) to assess overfitting or underfitting.
  * **residuals\_analysis.png:** Provides a scatter plot of Predicted Values vs. Residuals and a histogram of the error distribution to verify the assumption of normally distributed errors.

## Execution Instructions

### Step 1: Data Preparation

Ensure the following raw data files are located in the same directory as the Python scripts. These files contain the yield targets and weather features required for training.

  * `IA_CORN_raw.json` and `iowa.json`
  * `IL_SOYBEANS_raw.json` and `illinois.json`
  * `KS_WHEAT_raw.json` and `kansas.json`

### Step 2: Run Optimization Analysis

Execute the optimization script to generate the yield response curves and determine optimal climatic thresholds.

```bash
python ml_optimization.py
```

### Step 3: Run Model Diagnostics

Execute the diagnostics script to generate performance evaluation metrics and validation plots.

```bash
python ml_diagnostics.py
```

## Methodology and Implementation Details

  * **Feature Engineering:** Raw daily weather data is aggregated into "Growing Season" metrics specific to each crop's phenology (e.g., May-October for Corn and Soybeans; October-June cross-year aggregation for Winter Wheat).
  * **Model Architecture:** A Random Forest Regressor is employed with `n_estimators=100` and `max_depth=5`. The depth limit is imposed to prevent overfitting given the size of the historical dataset (2000-2025).
  * **Optimization Logic:** The script utilizes a simulation-based approach (ceteris paribus). It creates synthetic datasets where one variable (e.g., Temperature) is varied across a realistic range while holding other variables (Precipitation, Solar Radiation) constant at their historical means. The peak of the resulting prediction curve is identified as the optimal growing condition.

<!-- end list -->

```
```
