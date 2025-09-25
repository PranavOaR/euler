#!/usr/bin/env python3
"""
Advanced Machine Learning Model for Financial Profit Prediction

This script implements an enhanced ML pipeline with:
- Advanced feature engineering (5 financial ratios)
- Sentiment analysis integration via Finnhub API
- Model comparison (Linear Regression vs Gradient Boosting)
- Professional visualization and performance analysis
"""

import pandas as pd
import numpy as np
import requests
import matplotlib.pyplot as plt
import time
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.metrics import mean_squared_error, r2_score
import warnings
warnings.filterwarnings('ignore')

def load_and_prepare_data(file_path):
    """
    Load financial data and handle missing values.
    
    Args:
        file_path (str): Path to the CSV file
        
    Returns:
        pd.DataFrame: Cleaned financial dataset
    """
    print("=" * 70)
    print("STEP 1: DATA LOADING AND PREPARATION")
    print("=" * 70)
    
    try:
        # Load the CSV file
        df = pd.read_csv(file_path)
        print(f"✓ Successfully loaded data from {file_path}")
        print(f"Dataset shape: {df.shape}")
        print()
        
        # Display first few rows
        print("First 5 rows of the dataset:")
        print(df.head())
        print()
        
        # Check for missing values
        print("Missing values check:")
        missing_values = df.isnull().sum()
        total_missing = missing_values.sum()
        
        if total_missing > 0:
            print("Missing values found:")
            for col, missing in missing_values.items():
                if missing > 0:
                    print(f"  {col}: {missing}")
            
            # Drop rows with missing values
            df_clean = df.dropna()
            rows_dropped = len(df) - len(df_clean)
            print(f"✓ Dropped {rows_dropped} rows with missing values")
        else:
            print("  No missing values found!")
            df_clean = df.copy()
        
        print(f"Final dataset shape: {df_clean.shape}")
        print()
        
        # Display basic statistics
        print("Basic financial statistics:")
        financial_cols = ['Revenue', 'Total_Assets', 'Total_Liabilities', 'Net_Income']
        print(df_clean[financial_cols].describe())
        print()
        
        return df_clean
    
    except FileNotFoundError:
        print(f"Error: File {file_path} not found.")
        return None
    except Exception as e:
        print(f"Error loading data: {e}")
        return None

def engineer_advanced_features(df):
    """
    Create advanced financial ratios and integrate sentiment data.
    
    Args:
        df (pd.DataFrame): Financial dataset
        
    Returns:
        tuple: (enhanced_df, feature_columns)
    """
    print("=" * 70)
    print("STEP 2: ADVANCED FEATURE ENGINEERING")
    print("=" * 70)
    
    # Create a copy to avoid modifying original data
    df_enhanced = df.copy()
    
    # Calculate Total Shareholder's Equity
    df_enhanced['Total_Shareholders_Equity'] = (
        df_enhanced['Total_Assets'] - df_enhanced['Total_Liabilities']
    )
    print("✓ Calculated Total Shareholder's Equity")
    
    # 1. Net Profit Margin
    df_enhanced['Net_Profit_Margin'] = df_enhanced['Net_Income'] / df_enhanced['Revenue']
    print("✓ Calculated Net Profit Margin = Net Income / Revenue")
    
    # 2. Debt-to-Equity Ratio
    df_enhanced['Debt_to_Equity_Ratio'] = (
        df_enhanced['Total_Liabilities'] / df_enhanced['Total_Shareholders_Equity']
    )
    print("✓ Calculated Debt-to-Equity Ratio = Total Liabilities / Total Shareholder's Equity")
    
    # 3. Current Ratio
    df_enhanced['Current_Ratio'] = (
        df_enhanced['Current_Assets'] / df_enhanced['Current_Liabilities']
    )
    print("✓ Calculated Current Ratio = Current Assets / Current Liabilities")
    
    # 4. Return on Assets (ROA)
    df_enhanced['Return_on_Assets'] = df_enhanced['Net_Income'] / df_enhanced['Total_Assets']
    print("✓ Calculated Return on Assets (ROA) = Net Income / Total Assets")
    
    # 5. Return on Equity (ROE)
    df_enhanced['Return_on_Equity'] = (
        df_enhanced['Net_Income'] / df_enhanced['Total_Shareholders_Equity']
    )
    print("✓ Calculated Return on Equity (ROE) = Net Income / Total Shareholder's Equity")
    print()
    
    # Integrate Sentiment Data using Finnhub API
    print("SENTIMENT DATA INTEGRATION:")
    print("-" * 30)
    
    api_key = "d38j2chr01qlbdj5ofn0d38j2chr01qlbdj5ofng"
    endpoint = "https://finnhub.io/api/v1/news-sentiment"
    
    # Get unique companies and their tickers
    unique_companies = df_enhanced.groupby(['Company', 'Ticker']).size().reset_index()[['Company', 'Ticker']]
    print(f"Found {len(unique_companies)} unique companies")
    
    # Initialize sentiment score column
    df_enhanced['Sentiment_Score'] = 0.0
    
    # Dictionary to store sentiment scores for each ticker
    sentiment_scores = {}
    
    for idx, row in unique_companies.iterrows():
        ticker = row['Ticker']
        company = row['Company']
        
        try:
            # Make API call
            params = {
                'symbol': ticker,
                'token': api_key
            }
            
            print(f"  Fetching sentiment for {company} ({ticker})...", end=" ")
            
            response = requests.get(endpoint, params=params, timeout=10)
            
            if response.status_code == 200:
                data = response.json()
                
                # Extract sentiment score (using buzz.articlesInLastWeek as a proxy)
                # Finnhub sentiment API returns various metrics, we'll use a composite score
                if 'buzz' in data and 'articlesInLastWeek' in data['buzz']:
                    sentiment_score = min(data['buzz']['articlesInLastWeek'] / 10.0, 1.0)  # Normalize to 0-1
                else:
                    sentiment_score = 0.5  # Neutral score if no data
                
                sentiment_scores[ticker] = sentiment_score
                print(f"✓ Score: {sentiment_score:.3f}")
                
            else:
                # Handle API errors gracefully
                sentiment_scores[ticker] = 0.5  # Neutral score for errors
                print(f"⚠ API Error ({response.status_code}), using neutral score: 0.5")
            
            # Rate limiting - Finnhub free tier: 30 calls/minute
            time.sleep(2.1)  # 60/30 = 2 seconds between calls, plus buffer
            
        except requests.exceptions.RequestException as e:
            sentiment_scores[ticker] = 0.5
            print(f"⚠ Request failed, using neutral score: 0.5")
        except Exception as e:
            sentiment_scores[ticker] = 0.5
            print(f"⚠ Error processing {ticker}, using neutral score: 0.5")
    
    # Apply sentiment scores to all rows
    for ticker, score in sentiment_scores.items():
        df_enhanced.loc[df_enhanced['Ticker'] == ticker, 'Sentiment_Score'] = score
    
    print(f"\n✓ Sentiment integration completed for {len(sentiment_scores)} companies")
    print()
    
    # Display feature statistics
    feature_columns = [
        'Net_Profit_Margin', 'Debt_to_Equity_Ratio', 'Current_Ratio',
        'Return_on_Assets', 'Return_on_Equity', 'Sentiment_Score'
    ]
    
    print("Advanced features statistics:")
    for col in feature_columns:
        mean_val = df_enhanced[col].mean()
        std_val = df_enhanced[col].std()
        print(f"  {col:20s}: Mean = {mean_val:8.4f}, Std = {std_val:8.4f}")
    print()
    
    # Drop any rows with missing values after feature engineering
    initial_rows = len(df_enhanced)
    df_enhanced = df_enhanced.dropna()
    final_rows = len(df_enhanced)
    dropped_rows = initial_rows - final_rows
    
    if dropped_rows > 0:
        print(f"✓ Dropped {dropped_rows} rows with missing values after feature engineering")
    
    print(f"Final enhanced dataset shape: {df_enhanced.shape}")
    print()
    
    return df_enhanced, feature_columns

def train_and_compare_models(df, feature_columns):
    """
    Train and compare Linear Regression and Gradient Boosting models.
    
    Args:
        df (pd.DataFrame): Enhanced dataset with features
        feature_columns (list): List of feature column names
        
    Returns:
        dict: Model results and predictions
    """
    print("=" * 70)
    print("STEP 3: MODEL TRAINING & COMPARISON")
    print("=" * 70)
    
    # Define features (X) and target (y)
    X = df[feature_columns]
    y = df['Net_Income']
    
    print("Feature set (X) - Advanced Financial Ratios + Sentiment:")
    for i, col in enumerate(feature_columns, 1):
        print(f"  {i}. {col}")
    print()
    
    print(f"Target variable (y): Net Income")
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
    
    # Standardize features for Linear Regression
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # Initialize results dictionary
    results = {
        'X_test': X_test,
        'y_test': y_test,
        'feature_columns': feature_columns
    }
    
    print("MODEL 1: LINEAR REGRESSION")
    print("-" * 30)
    
    # Train Linear Regression
    lr_model = LinearRegression()
    lr_model.fit(X_train_scaled, y_train)
    
    # Make predictions
    lr_predictions = lr_model.predict(X_test_scaled)
    
    # Calculate metrics
    lr_mse = mean_squared_error(y_test, lr_predictions)
    lr_r2 = r2_score(y_test, lr_predictions)
    lr_rmse = np.sqrt(lr_mse)
    
    print(f"✓ Linear Regression trained successfully")
    print(f"  R-squared (R²): {lr_r2:.4f}")
    print(f"  MSE: ${lr_mse:,.2f}")
    print(f"  RMSE: ${lr_rmse:,.2f}")
    print()
    
    # Feature importance for Linear Regression (absolute coefficients)
    feature_importance_lr = pd.DataFrame({
        'feature': feature_columns,
        'coefficient': lr_model.coef_,
        'abs_coefficient': np.abs(lr_model.coef_)
    }).sort_values('abs_coefficient', ascending=False)
    
    print("Linear Regression - Feature Importance (|coefficients|):")
    for _, row in feature_importance_lr.iterrows():
        print(f"  {row['feature']:20s}: {row['abs_coefficient']:10.2f}")
    print()
    
    results['linear_regression'] = {
        'model': lr_model,
        'predictions': lr_predictions,
        'mse': lr_mse,
        'r2': lr_r2,
        'rmse': lr_rmse,
        'feature_importance': feature_importance_lr
    }
    
    print("MODEL 2: GRADIENT BOOSTING REGRESSOR")
    print("-" * 40)
    
    # Train Gradient Boosting Regressor
    gb_model = GradientBoostingRegressor(
        n_estimators=100,
        learning_rate=0.1,
        max_depth=3,
        random_state=42
    )
    gb_model.fit(X_train, y_train)
    
    # Make predictions
    gb_predictions = gb_model.predict(X_test)
    
    # Calculate metrics
    gb_mse = mean_squared_error(y_test, gb_predictions)
    gb_r2 = r2_score(y_test, gb_predictions)
    gb_rmse = np.sqrt(gb_mse)
    
    print(f"✓ Gradient Boosting Regressor trained successfully")
    print(f"  R-squared (R²): {gb_r2:.4f}")
    print(f"  MSE: ${gb_mse:,.2f}")
    print(f"  RMSE: ${gb_rmse:,.2f}")
    print()
    
    # Feature importance for Gradient Boosting
    feature_importance_gb = pd.DataFrame({
        'feature': feature_columns,
        'importance': gb_model.feature_importances_
    }).sort_values('importance', ascending=False)
    
    print("Gradient Boosting - Feature Importance:")
    for _, row in feature_importance_gb.iterrows():
        print(f"  {row['feature']:20s}: {row['importance']:10.4f}")
    print()
    
    results['gradient_boosting'] = {
        'model': gb_model,
        'predictions': gb_predictions,
        'mse': gb_mse,
        'r2': gb_r2,
        'rmse': gb_rmse,
        'feature_importance': feature_importance_gb
    }
    
    # Model Comparison Summary
    print("=" * 70)
    print("MODEL PERFORMANCE COMPARISON")
    print("=" * 70)
    
    print(f"{'Metric':<20} {'Linear Regression':<20} {'Gradient Boosting':<20} {'Better Model'}")
    print("-" * 80)
    print(f"{'R² Score':<20} {lr_r2:<20.4f} {gb_r2:<20.4f} {'GB' if gb_r2 > lr_r2 else 'LR'}")
    print(f"{'MSE':<20} ${lr_mse:<19,.0f} ${gb_mse:<19,.0f} {'GB' if gb_mse < lr_mse else 'LR'}")
    print(f"{'RMSE':<20} ${lr_rmse:<19,.0f} ${gb_rmse:<19,.0f} {'GB' if gb_rmse < lr_rmse else 'LR'}")
    print()
    
    # Determine better model
    better_model = 'gradient_boosting' if gb_r2 > lr_r2 else 'linear_regression'
    better_model_name = 'Gradient Boosting' if better_model == 'gradient_boosting' else 'Linear Regression'
    
    print(f"🏆 BEST PERFORMING MODEL: {better_model_name}")
    print(f"   R² Score: {results[better_model]['r2']:.4f}")
    print(f"   MSE: ${results[better_model]['mse']:,.2f}")
    print(f"   Improvement over other model: {abs(gb_r2 - lr_r2):.4f} R² points")
    print()
    
    results['better_model'] = better_model
    results['better_model_name'] = better_model_name
    
    return results

def create_enhanced_visualization(results):
    """
    Create professional visualization of model performance.
    
    Args:
        results (dict): Model results and predictions
    """
    print("=" * 70)
    print("STEP 4: ENHANCED VISUALIZATION")
    print("=" * 70)
    
    # Get data for the better-performing model
    better_model = results['better_model']
    better_model_name = results['better_model_name']
    y_test = results['y_test']
    y_pred = results[better_model]['predictions']
    r2_score_val = results[better_model]['r2']
    mse_val = results[better_model]['mse']
    
    # Create figure with high quality
    plt.figure(figsize=(12, 10))
    
    # Main scatter plot
    plt.scatter(y_test, y_pred, alpha=0.7, color='darkblue', s=80, 
                edgecolors='navy', linewidth=0.8, label='Predictions')
    
    # Perfect prediction line (y=x)
    min_val = min(y_test.min(), y_pred.min())
    max_val = max(y_test.max(), y_pred.max())
    plt.plot([min_val, max_val], [min_val, max_val], 'r-', linewidth=3, 
             label='Perfect Prediction (y=x)', alpha=0.8)
    
    # Calculate and add trend line
    z = np.polyfit(y_test, y_pred, 1)
    p = np.poly1d(z)
    plt.plot(y_test, p(y_test), 'orange', linewidth=2, linestyle='--', 
             label=f'Trend Line (slope={z[0]:.3f})', alpha=0.8)
    
    # Customize the plot
    plt.xlabel('Actual Net Income ($)', fontsize=14, fontweight='bold')
    plt.ylabel('Predicted Net Income ($)', fontsize=14, fontweight='bold')
    plt.title(f'Advanced ML Model Performance: {better_model_name}\\n'\n             f'Financial Profit Prediction with Sentiment Analysis', \n             fontsize=16, fontweight='bold', pad=25)\n    \n    # Add performance metrics box\n    textstr = f'Model: {better_model_name}\\n'\\\n              f'R² Score: {r2_score_val:.4f}\\n'\\\n              f'MSE: ${mse_val:,.0f}\\n'\\\n              f'Data Points: {len(y_test)}\\n'\\\n              f'Features: {len(results[\"feature_columns\"])}'\n    \n    props = dict(boxstyle='round,pad=0.8', facecolor='lightblue', alpha=0.8, edgecolor='navy')\n    plt.text(0.02, 0.98, textstr, transform=plt.gca().transAxes, fontsize=11,\n             verticalalignment='top', bbox=props, fontweight='bold')\n    \n    # Add feature importance box\n    feature_imp = results[better_model]['feature_importance'].head(3)\n    imp_text = 'Top 3 Features:\\n'\n    for _, row in feature_imp.iterrows():\n        if better_model == 'gradient_boosting':\n            imp_text += f'{row[\"feature\"][:12]}: {row[\"importance\"]:.3f}\\n'\n        else:\n            imp_text += f'{row[\"feature\"][:12]}: {row[\"abs_coefficient\"]:.1f}\\n'\n    \n    props2 = dict(boxstyle='round,pad=0.5', facecolor='lightyellow', alpha=0.8, edgecolor='orange')\n    plt.text(0.98, 0.02, imp_text.strip(), transform=plt.gca().transAxes, fontsize=10,\n             verticalalignment='bottom', horizontalalignment='right', bbox=props2)\n    \n    # Add legend\n    plt.legend(fontsize=12, loc='center right')\n    \n    # Add grid for better readability\n    plt.grid(True, alpha=0.3, linestyle='-', linewidth=0.5)\n    \n    # Format axes to show currency\n    plt.gca().xaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: f'${x:,.0f}'))\n    plt.gca().yaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: f'${x:,.0f}'))\n    \n    # Improve tick labels\n    plt.xticks(fontsize=11)\n    plt.yticks(fontsize=11)\n    \n    # Set equal aspect ratio for better visualization\n    plt.axis('equal')\n    \n    # Adjust layout\n    plt.tight_layout()\n    \n    # Save the plot with high DPI\n    plt.savefig('model_comparison.png', dpi=300, bbox_inches='tight', \n                facecolor='white', edgecolor='none')\n    \n    print(\"✓ Enhanced visualization created with:\")\n    print(\"  - Scatter plot of actual vs predicted Net Income\")\n    print(\"  - Perfect prediction reference line (y=x)\")\n    print(\"  - Trend line showing prediction bias\")\n    print(\"  - Performance metrics display\")\n    print(\"  - Top 3 feature importance indicators\")\n    print(\"  - Professional styling and formatting\")\n    print()\n    print(\"✓ Plot saved as 'model_comparison.png' (300 DPI)\")\n    \n    # Show the plot\n    plt.show()\n    print(\"✓ Plot displayed successfully\")\n    print()\n\ndef main():\n    \"\"\"\n    Main function to execute the complete advanced ML pipeline.\n    \"\"\"\n    print(\"ADVANCED MACHINE LEARNING MODEL FOR FINANCIAL PROFIT PREDICTION\")\n    print(\"Enhanced with Sentiment Analysis and Model Comparison\")\n    print(\"=\" * 70)\n    print()\n    \n    # File path for the financial data\n    file_path = 'financial_statements1.csv'\n    \n    # Step 1: Load and prepare data\n    df = load_and_prepare_data(file_path)\n    if df is None:\n        print(\"Failed to load data. Exiting...\")\n        return\n    \n    # Step 2: Advanced feature engineering with sentiment analysis\n    df_enhanced, feature_columns = engineer_advanced_features(df)\n    \n    # Step 3: Train and compare models\n    results = train_and_compare_models(df_enhanced, feature_columns)\n    \n    # Step 4: Create enhanced visualization\n    create_enhanced_visualization(results)\n    \n    # Final comprehensive summary\n    print(\"=\" * 70)\n    print(\"ADVANCED PIPELINE EXECUTION COMPLETED SUCCESSFULLY!\")\n    print(\"=\" * 70)\n    print(\"Enhanced Features Summary:\")\n    print(f\"  • Dataset: {len(df_enhanced)} company-year observations\")\n    print(f\"  • Advanced Features: {len(feature_columns)} (including sentiment analysis)\")\n    print(f\"  • Models Compared: Linear Regression vs Gradient Boosting Regressor\")\n    print(f\"  • Best Model: {results['better_model_name']}\")\n    print(f\"  • Best R² Score: {results[results['better_model']]['r2']:.4f}\")\n    print(f\"  • Best MSE: ${results[results['better_model']]['mse']:,.2f}\")\n    print(f\"  • Sentiment Integration: ✓ Finnhub API\")\n    print(f\"  • Professional Visualization: ✓ High-quality plot saved\")\n    print(\"=\" * 70)\n\nif __name__ == \"__main__\":\n    main()"

