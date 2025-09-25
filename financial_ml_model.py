#!/usr/bin/env python3
"""
Machine Learning Model for Predicting Company Net Profit Based on Financial Ratios

This script implements a complete ML pipeline including:
1. Data loading and exploration
2. Feature engineering (financial ratios)
3. Data preprocessing
4. Model training with GradientBoostingRegressor
5. Model evaluation with MSE and R2 metrics
6. Visualization of results
"""

# Step 1: Setup and Data Loading
# Import necessary libraries
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.metrics import mean_squared_error, r2_score
import matplotlib.pyplot as plt
import os

def load_and_explore_data(file_path):
    """Load the financial data and display basic information."""
    print("=" * 60)
    print("STEP 1: SETUP AND DATA LOADING")
    print("=" * 60)
    
    # Load the CSV file
    try:
        df = pd.read_csv(file_path)
        print(f"✓ Successfully loaded data from {file_path}")
        print(f"Dataset shape: {df.shape}")
        print()
        
        # Display first 5 rows
        print("First 5 rows of the dataset:")
        print(df.head())
        print()
        
        # Display columns
        print("Columns in the dataset:")
        for i, col in enumerate(df.columns, 1):
            print(f"{i:2d}. {col}")
        print()
        
        # Basic statistics
        print("Basic dataset statistics:")
        print(df.describe())
        print()
        
        return df
    
    except FileNotFoundError:
        print(f"Error: File {file_path} not found.")
        return None
    except Exception as e:
        print(f"Error loading data: {e}")
        return None

def engineer_features(df):
    """Create financial ratio features."""
    print("=" * 60)
    print("STEP 2: FEATURE ENGINEERING - CALCULATING FINANCIAL RATIOS")
    print("=" * 60)
    
    # Create a copy to avoid modifying original data
    df_engineered = df.copy()
    
    # Calculate Total Shareholder's Equity
    df_engineered['Total_Shareholders_Equity'] = df_engineered['Total Assets'] - df_engineered['Total Liabilities']
    print("✓ Calculated Total Shareholder's Equity = Total Assets - Total Liabilities")
    
    # 1. Net Profit Margin = Net Income / Revenue
    df_engineered['Net_Profit_Margin'] = df_engineered['Net Income'] / df_engineered['Revenue']
    print("✓ Calculated Net Profit Margin = Net Income / Revenue")
    
    # 2. Debt-to-Equity Ratio = Total Liabilities / Total Shareholder's Equity
    df_engineered['Debt_to_Equity_Ratio'] = df_engineered['Total Liabilities'] / df_engineered['Total_Shareholders_Equity']
    print("✓ Calculated Debt-to-Equity Ratio = Total Liabilities / Total Shareholder's Equity")
    
    # 3. Current Ratio = Current Assets / Current Liabilities
    df_engineered['Current_Ratio'] = df_engineered['Current Assets'] / df_engineered['Current Liabilities']
    print("✓ Calculated Current Ratio = Current Assets / Current Liabilities")
    
    print()
    print("New financial ratios created:")
    ratio_columns = ['Net_Profit_Margin', 'Debt_to_Equity_Ratio', 'Current_Ratio']
    for col in ratio_columns:
        print(f"  - {col}: Mean = {df_engineered[col].mean():.4f}, Std = {df_engineered[col].std():.4f}")
    print()
    
    return df_engineered, ratio_columns

def clean_and_preprocess_data(df, ratio_columns):
    """Clean data and prepare features and target variables."""
    print("=" * 60)
    print("STEP 3: DATA CLEANING AND PREPROCESSING")
    print("=" * 60)
    
    # Check for missing values
    print("Missing values before cleaning:")
    missing_values = df.isnull().sum()
    for col, missing in missing_values.items():
        if missing > 0:
            print(f"  {col}: {missing}")
    
    if missing_values.sum() == 0:
        print("  No missing values found!")
    print()
    
    # Drop rows with NaN values created by ratio calculations
    df_clean = df.dropna()
    rows_dropped = len(df) - len(df_clean)
    print(f"✓ Dropped {rows_dropped} rows with missing values")
    print(f"Dataset shape after cleaning: {df_clean.shape}")
    print()
    
    # Define feature set (X) - financial ratios
    X = df_clean[ratio_columns]
    print("Feature set (X) - Financial Ratios:")
    for i, col in enumerate(ratio_columns, 1):
        print(f"  {i}. {col}")
    print()
    
    # Define target variable (y) - Net Income
    y = df_clean['Net Income']
    print("Target variable (y): Net Income")
    print(f"Target statistics: Mean = ${y.mean():,.2f}, Std = ${y.std():,.2f}")
    print()
    
    # Split data into training and testing sets (80-20 split)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, shuffle=True
    )
    
    print("Data split completed:")
    print(f"  Training set: {X_train.shape[0]} samples ({X_train.shape[0]/len(X):.1%})")
    print(f"  Testing set:  {X_test.shape[0]} samples ({X_test.shape[0]/len(X):.1%})")
    print()
    
    return X_train, X_test, y_train, y_test, df_clean

def train_model(X_train, y_train):
    """Train the Gradient Boosting Regressor model."""
    print("=" * 60)
    print("STEP 4: MODEL TRAINING")
    print("=" * 60)
    
    # Instantiate the model
    model = GradientBoostingRegressor(
        n_estimators=100,
        learning_rate=0.1,
        max_depth=3,
        random_state=42
    )
    
    print("✓ Instantiated GradientBoostingRegressor with parameters:")
    print(f"  - n_estimators: {model.n_estimators}")
    print(f"  - learning_rate: {model.learning_rate}")
    print(f"  - max_depth: {model.max_depth}")
    print(f"  - random_state: {model.random_state}")
    print()
    
    # Train the model
    print("Training the model...")
    model.fit(X_train, y_train)
    print("✓ Model has been successfully trained!")
    print()
    
    # Display feature importance
    feature_importance = pd.DataFrame({
        'feature': X_train.columns,
        'importance': model.feature_importances_
    }).sort_values('importance', ascending=False)
    
    print("Feature Importance:")
    for _, row in feature_importance.iterrows():
        print(f"  {row['feature']:20s}: {row['importance']:.4f}")
    print()
    
    return model

def evaluate_model(model, X_test, y_test):
    """Evaluate the model and calculate performance metrics."""
    print("=" * 60)
    print("STEP 5: MODEL EVALUATION AND PERFORMANCE METRICS")
    print("=" * 60)
    
    # Make predictions on test set
    y_pred = model.predict(X_test)
    print("✓ Generated predictions on test set")
    print()
    
    # Calculate Mean Squared Error (MSE)
    mse = mean_squared_error(y_test, y_pred)
    print(f"Mean Squared Error (MSE): ${mse:,.2f}")
    print("  → MSE measures the average squared difference between actual and predicted values.")
    print("  → Lower MSE indicates better model performance.")
    print()
    
    # Calculate R-squared (R2) score
    r2 = r2_score(y_test, y_pred)
    print(f"R-squared (R²) Score: {r2:.4f}")
    print("  → R² measures the proportion of variance in the target variable explained by the model.")
    print("  → R² ranges from 0 to 1, where 1 indicates perfect prediction.")
    print(f"  → This model explains {r2:.1%} of the variance in Net Income.")
    print()
    
    # Calculate additional metrics
    rmse = np.sqrt(mse)
    mae = np.mean(np.abs(y_test - y_pred))
    
    print("Additional Performance Metrics:")
    print(f"  Root Mean Squared Error (RMSE): ${rmse:,.2f}")
    print(f"  Mean Absolute Error (MAE):      ${mae:,.2f}")
    print()
    
    # Prediction statistics
    print("Prediction Analysis:")
    print(f"  Actual values range:    ${y_test.min():,.2f} to ${y_test.max():,.2f}")
    print(f"  Predicted values range: ${y_pred.min():,.2f} to ${y_pred.max():,.2f}")
    print(f"  Mean actual value:      ${y_test.mean():,.2f}")
    print(f"  Mean predicted value:   ${y_pred.mean():,.2f}")
    print()
    
    return y_pred, mse, r2

def create_visualization(y_test, y_pred, r2, mse):
    """Create visualization of model performance."""
    print("=" * 60)
    print("STEP 6: VISUALIZATION")
    print("=" * 60)
    
    # Create figure and axis
    plt.figure(figsize=(10, 8))
    
    # Create scatter plot
    plt.scatter(y_test, y_pred, alpha=0.7, color='blue', s=60, edgecolors='black', linewidth=0.5)
    
    # Add diagonal line (y=x) for perfect prediction reference
    min_val = min(y_test.min(), y_pred.min())
    max_val = max(y_test.max(), y_pred.max())
    plt.plot([min_val, max_val], [min_val, max_val], 'r--', linewidth=2, label='Perfect Prediction (y=x)')
    
    # Customize the plot
    plt.xlabel('Actual Net Income ($)', fontsize=12, fontweight='bold')
    plt.ylabel('Predicted Net Income ($)', fontsize=12, fontweight='bold')
    plt.title('Model Performance: Actual vs Predicted Net Income\nGradient Boosting Regressor', 
              fontsize=14, fontweight='bold', pad=20)
    
    # Add performance metrics to the plot
    textstr = f'R² Score: {r2:.4f}\nMSE: ${mse:,.0f}'
    props = dict(boxstyle='round', facecolor='wheat', alpha=0.8)
    plt.text(0.05, 0.95, textstr, transform=plt.gca().transAxes, fontsize=11,
             verticalalignment='top', bbox=props)
    
    # Add legend
    plt.legend(fontsize=10)
    
    # Add grid for better readability
    plt.grid(True, alpha=0.3)
    
    # Format axes to show currency
    plt.gca().xaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: f'${x:,.0f}'))
    plt.gca().yaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: f'${x:,.0f}'))
    
    # Adjust layout
    plt.tight_layout()
    
    print("✓ Created scatter plot visualization")
    print("  - X-axis: Actual Net Income values from test set")
    print("  - Y-axis: Model's predicted Net Income values")
    print("  - Red dashed line: Perfect prediction line (y=x)")
    print("  - Points close to the red line indicate accurate predictions")
    print()
    
    # Save the plot
    plt.savefig('model_performance_plot.png', dpi=300, bbox_inches='tight')
    print("✓ Plot saved as 'model_performance_plot.png'")
    
    # Show the plot
    plt.show()
    print("✓ Plot displayed successfully")
    print()

def main():
    """Main function to execute the complete ML pipeline."""
    print("MACHINE LEARNING MODEL FOR NET PROFIT PREDICTION")
    print("Based on Financial Ratios Analysis")
    print("=" * 60)
    print()
    
    # File path for the financial data
    file_path = 'financial_data.csv'
    
    # Step 1: Load and explore data
    df = load_and_explore_data(file_path)
    if df is None:
        print("Failed to load data. Exiting...")
        return
    
    # Step 2: Feature engineering
    df_engineered, ratio_columns = engineer_features(df)
    
    # Step 3: Data cleaning and preprocessing
    X_train, X_test, y_train, y_test, df_clean = clean_and_preprocess_data(df_engineered, ratio_columns)
    
    # Step 4: Model training
    model = train_model(X_train, y_train)
    
    # Step 5: Model evaluation
    y_pred, mse, r2 = evaluate_model(model, X_test, y_test)
    
    # Step 6: Visualization
    create_visualization(y_test, y_pred, r2, mse)
    
    # Final summary
    print("=" * 60)
    print("PIPELINE EXECUTION COMPLETED SUCCESSFULLY!")
    print("=" * 60)
    print("Summary:")
    print(f"  • Dataset: {len(df_clean)} companies with {len(ratio_columns)} financial ratios")
    print(f"  • Model: Gradient Boosting Regressor")
    print(f"  • Performance: R² = {r2:.4f}, MSE = ${mse:,.2f}")
    print(f"  • The model explains {r2:.1%} of the variance in Net Income")
    print("=" * 60)

if __name__ == "__main__":
    main()
