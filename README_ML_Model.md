# Financial Net Profit Prediction Model

This project implements a machine learning model to predict a company's net profit based on its financial ratios using a Gradient Boosting Regressor.

## 🎯 Project Overview

The model analyzes financial data and creates engineered features (financial ratios) to predict net income. It includes a complete ML pipeline with data loading, feature engineering, model training, evaluation, and visualization.

## 📁 Files Generated

- `financial_data.csv` - Sample dataset with financial statements of 30 companies
- `financial_ml_model.py` - Complete ML pipeline implementation
- `model_performance_plot.png` - Visualization of actual vs predicted values
- `requirements.txt` - Python dependencies

## 🚀 How to Run

1. **Install Dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

2. **Run the Model:**
   ```bash
   python financial_ml_model.py
   ```
   or
   ```bash
   py financial_ml_model.py
   ```

## 📊 Features and Pipeline

### 1. Data Loading
- Loads financial data from `financial_data.csv`
- Displays dataset structure and basic statistics

### 2. Feature Engineering
Creates three key financial ratios:
- **Net Profit Margin** = Net Income / Revenue
- **Debt-to-Equity Ratio** = Total Liabilities / Total Shareholder's Equity
- **Current Ratio** = Current Assets / Current Liabilities

### 3. Data Preprocessing
- Handles missing values
- Splits data into training (80%) and testing (20%) sets
- Prepares features (X) and target variable (y)

### 4. Model Training
- Uses `GradientBoostingRegressor` from scikit-learn
- Displays feature importance after training

### 5. Model Evaluation
Calculates key performance metrics:
- **Mean Squared Error (MSE)** - Average squared prediction error
- **R-squared (R²)** - Proportion of variance explained by the model
- **Root Mean Squared Error (RMSE)**
- **Mean Absolute Error (MAE)**

### 6. Visualization
- Creates scatter plot of actual vs predicted values
- Includes perfect prediction line (y=x) for reference
- Saves plot as `model_performance_plot.png`

## 📈 Model Performance

The current model demonstrates the complete ML pipeline. Key insights:

- **Dataset**: 30 companies with 3 engineered financial ratios
- **Model**: Gradient Boosting Regressor
- **Feature Importance**: Current Ratio is the most important predictor
- **Visualization**: Shows prediction accuracy through scatter plot

## 🔧 Customization Options

### Modify the Dataset
Replace `financial_data.csv` with your own data. Required columns:
- Company Name
- Revenue
- Total Assets
- Total Liabilities
- Current Assets
- Current Liabilities
- Net Income

### Adjust Model Parameters
In `financial_ml_model.py`, modify the GradientBoostingRegressor parameters:
```python
model = GradientBoostingRegressor(
    n_estimators=100,      # Number of boosting stages
    learning_rate=0.1,     # Learning rate
    max_depth=3,           # Maximum depth of trees
    random_state=42        # Random seed for reproducibility
)
```

### Add More Financial Ratios
Extend the feature engineering section to include additional ratios:
- Return on Assets (ROA)
- Return on Equity (ROE)
- Quick Ratio
- Asset Turnover Ratio

## 📚 Dependencies

- **pandas**: Data manipulation and analysis
- **numpy**: Numerical computing
- **scikit-learn**: Machine learning library
- **matplotlib**: Data visualization

## 💡 Future Enhancements

1. **Cross-Validation**: Implement k-fold cross-validation for better model assessment
2. **Feature Selection**: Use statistical methods to select the most predictive features
3. **Hyperparameter Tuning**: Use GridSearchCV to optimize model parameters
4. **Multiple Models**: Compare different algorithms (Random Forest, SVR, Neural Networks)
5. **Time Series Analysis**: Include temporal trends in financial performance
6. **Industry Segmentation**: Build separate models for different industry sectors

## 🎨 Output Examples

The model generates:
- Detailed console output with step-by-step progress
- Performance metrics and model evaluation
- Feature importance rankings
- High-quality scatter plot visualization
- Summary statistics and insights

This implementation provides a robust foundation for financial prediction modeling and can be easily extended for production use cases.
