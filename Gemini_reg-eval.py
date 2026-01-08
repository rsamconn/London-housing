"""
Comprehensive Evaluation Script for Linear Regression Models
=============================================================

This script evaluates fitted sklearn regression models (LinearRegression, 
Lasso, Ridge) using multiple metrics and diagnostic plots.

Usage:
    Assumes you have:
    - reg_model: fitted sklearn regression model
    - X_train, y_train: training data
    - X_test, y_test: test data
    
    Then run: evaluate_regression_model(reg_model, X_train, y_train, X_test, y_test)
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.model_selection import cross_val_score, cross_validate
from scipy import stats
import warnings
warnings.filterwarnings('ignore')

# Set visualization style
sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (12, 6)


def evaluate_regression_model(reg_model, X_train, y_train, X_test, y_test, 
                               cv_folds=5, save_plots=True):
    """
    Comprehensive evaluation of a fitted regression model.
    
    Parameters:
    -----------
    reg_model : sklearn estimator
        Fitted regression model (LinearRegression, Lasso, or Ridge)
    X_train : array-like
        Training features
    y_train : array-like
        Training target values
    X_test : array-like
        Test features
    y_test : array-like
        Test target values
    cv_folds : int, default=5
        Number of cross-validation folds
    save_plots : bool, default=True
        Whether to save diagnostic plots
        
    Returns:
    --------
    dict : Dictionary containing all evaluation metrics
    """
    
    print("="*80)
    print("COMPREHENSIVE REGRESSION MODEL EVALUATION")
    print("="*80)
    print(f"\nModel Type: {type(reg_model).__name__}")
    print(f"Training Set Size: {X_train.shape[0]} samples, {X_train.shape[1]} features")
    print(f"Test Set Size: {X_test.shape[0]} samples")
    
    # Generate predictions
    y_train_pred = reg_model.predict(X_train)
    y_test_pred = reg_model.predict(X_test)
    
    # Initialize results dictionary
    results = {}
    
    # ========================================================================
    # 1. R² SCORE (COEFFICIENT OF DETERMINATION)
    # ========================================================================
    print("\n" + "="*80)
    print("1. R² SCORE (Coefficient of Determination)")
    print("="*80)
    
    train_r2 = r2_score(y_train, y_train_pred)
    test_r2 = r2_score(y_test, y_test_pred)
    
    results['train_r2'] = train_r2
    results['test_r2'] = test_r2
    
    print(f"\nTraining R²:   {train_r2:.4f}")
    print(f"Test R²:       {test_r2:.4f}")
    print(f"Difference:    {train_r2 - test_r2:.4f}")
    
    if test_r2 < 0:
        print("⚠ WARNING: Negative R² indicates model performs worse than baseline!")
    elif test_r2 < 0.5:
        print("⚠ Model explains less than 50% of variance - consider feature engineering")
    elif test_r2 > 0.8:
        print("✓ Strong model - explains >80% of variance")
    
    if (train_r2 - test_r2) > 0.1:
        print("⚠ Large gap suggests overfitting")
    
    # ========================================================================
    # 2. ADJUSTED R²
    # ========================================================================
    print("\n" + "="*80)
    print("2. ADJUSTED R²")
    print("="*80)
    
    n_train = X_train.shape[0]
    n_test = X_test.shape[0]
    p = X_train.shape[1]
    
    train_adj_r2 = 1 - (1 - train_r2) * (n_train - 1) / (n_train - p - 1)
    test_adj_r2 = 1 - (1 - test_r2) * (n_test - 1) / (n_test - p - 1)
    
    results['train_adj_r2'] = train_adj_r2
    results['test_adj_r2'] = test_adj_r2
    
    print(f"\nTraining Adjusted R²:  {train_adj_r2:.4f}")
    print(f"Test Adjusted R²:      {test_adj_r2:.4f}")
    print(f"\nAdjusted R² penalizes for {p} features")
    
    # ========================================================================
    # 3. RMSE (ROOT MEAN SQUARED ERROR)
    # ========================================================================
    print("\n" + "="*80)
    print("3. RMSE (Root Mean Squared Error)")
    print("="*80)
    
    train_rmse = np.sqrt(mean_squared_error(y_train, y_train_pred))
    test_rmse = np.sqrt(mean_squared_error(y_test, y_test_pred))
    
    results['train_rmse'] = train_rmse
    results['test_rmse'] = test_rmse
    
    # Calculate as percentage of mean price
    train_rmse_pct = (train_rmse / np.mean(y_train)) * 100
    test_rmse_pct = (test_rmse / np.mean(y_test)) * 100
    
    print(f"\nTraining RMSE: £{train_rmse:,.2f}")
    print(f"Test RMSE:     £{test_rmse:,.2f}")
    print(f"\nAs percentage of mean price:")
    print(f"Training:      {train_rmse_pct:.2f}%")
    print(f"Test:          {test_rmse_pct:.2f}%")
    
    if test_rmse / train_rmse > 1.5:
        print("⚠ Test RMSE is >1.5x training RMSE - significant overfitting")
    
    # ========================================================================
    # 4. MAE (MEAN ABSOLUTE ERROR)
    # ========================================================================
    print("\n" + "="*80)
    print("4. MAE (Mean Absolute Error)")
    print("="*80)
    
    train_mae = mean_absolute_error(y_train, y_train_pred)
    test_mae = mean_absolute_error(y_test, y_test_pred)
    
    results['train_mae'] = train_mae
    results['test_mae'] = test_mae
    
    train_mae_pct = (train_mae / np.mean(y_train)) * 100
    test_mae_pct = (test_mae / np.mean(y_test)) * 100
    
    print(f"\nTraining MAE:  £{train_mae:,.2f}")
    print(f"Test MAE:      £{test_mae:,.2f}")
    print(f"\nAs percentage of mean price:")
    print(f"Training:      {train_mae_pct:.2f}%")
    print(f"Test:          {test_mae_pct:.2f}%")
    
    print(f"\nInterpretation: Typical prediction error is £{test_mae:,.2f}")
    
    # ========================================================================
    # 5. MAPE (MEAN ABSOLUTE PERCENTAGE ERROR)
    # ========================================================================
    print("\n" + "="*80)
    print("5. MAPE (Mean Absolute Percentage Error)")
    print("="*80)
    
    # Avoid division by zero
    train_mape = np.mean(np.abs((y_train - y_train_pred) / y_train)) * 100
    test_mape = np.mean(np.abs((y_test - y_test_pred) / y_test)) * 100
    
    results['train_mape'] = train_mape
    results['test_mape'] = test_mape
    
    print(f"\nTraining MAPE: {train_mape:.2f}%")
    print(f"Test MAPE:     {test_mape:.2f}%")
    print("\n⚠ Note: MAPE can be biased toward lower-priced properties")
    
    # ========================================================================
    # 6. CROSS-VALIDATION SCORES
    # ========================================================================
    print("\n" + "="*80)
    print(f"6. CROSS-VALIDATION ({cv_folds}-Fold)")
    print("="*80)
    
    print(f"\nPerforming {cv_folds}-fold cross-validation on training data...")
    
    # Perform cross-validation with multiple metrics
    cv_results = cross_validate(
        reg_model, X_train, y_train, 
        cv=cv_folds,
        scoring={
            'r2': 'r2',
            'neg_rmse': 'neg_root_mean_squared_error',
            'neg_mae': 'neg_mean_absolute_error'
        },
        return_train_score=True
    )
    
    cv_r2_mean = cv_results['test_r2'].mean()
    cv_r2_std = cv_results['test_r2'].std()
    cv_rmse_mean = -cv_results['test_neg_rmse'].mean()
    cv_rmse_std = cv_results['test_neg_rmse'].std()
    cv_mae_mean = -cv_results['test_neg_mae'].mean()
    cv_mae_std = cv_results['test_neg_mae'].std()
    
    results['cv_r2_mean'] = cv_r2_mean
    results['cv_r2_std'] = cv_r2_std
    results['cv_rmse_mean'] = cv_rmse_mean
    results['cv_rmse_std'] = cv_rmse_std
    results['cv_mae_mean'] = cv_mae_mean
    results['cv_mae_std'] = cv_mae_std
    
    print(f"\nCross-Validated R²:    {cv_r2_mean:.4f} (±{cv_r2_std:.4f})")
    print(f"Cross-Validated RMSE:  £{cv_rmse_mean:,.2f} (±£{cv_rmse_std:,.2f})")
    print(f"Cross-Validated MAE:   £{cv_mae_mean:,.2f} (±£{cv_mae_std:,.2f})")
    
    if cv_r2_std > 0.1:
        print("⚠ High standard deviation - model performance varies significantly across folds")
    else:
        print("✓ Low standard deviation - model performance is stable")
    
    # ========================================================================
    # 7. RESIDUAL ANALYSIS
    # ========================================================================
    print("\n" + "="*80)
    print("7. RESIDUAL ANALYSIS")
    print("="*80)
    
    train_residuals = y_train - y_train_pred
    test_residuals = y_test - y_test_pred
    
    print("\nTraining Residuals:")
    print(f"  Mean:     £{np.mean(train_residuals):,.2f} (should be ~0)")
    print(f"  Std Dev:  £{np.std(train_residuals):,.2f}")
    print(f"  Min:      £{np.min(train_residuals):,.2f}")
    print(f"  Max:      £{np.max(train_residuals):,.2f}")
    
    print("\nTest Residuals:")
    print(f"  Mean:     £{np.mean(test_residuals):,.2f} (should be ~0)")
    print(f"  Std Dev:  £{np.std(test_residuals):,.2f}")
    print(f"  Min:      £{np.min(test_residuals):,.2f}")
    print(f"  Max:      £{np.max(test_residuals):,.2f}")
    
    # Test for normality (Shapiro-Wilk test on sample if large dataset)
    if len(test_residuals) > 5000:
        sample_residuals = np.random.choice(test_residuals, 5000, replace=False)
    else:
        sample_residuals = test_residuals
    
    _, p_value = stats.shapiro(sample_residuals)
    print(f"\nShapiro-Wilk Normality Test (p-value): {p_value:.4f}")
    if p_value < 0.05:
        print("⚠ Residuals are NOT normally distributed (p < 0.05)")
    else:
        print("✓ Residuals appear normally distributed (p >= 0.05)")
    
    # Create comprehensive residual plots
    fig = plt.figure(figsize=(16, 12))
    
    # 1. Residuals vs Predicted (Test Set)
    ax1 = plt.subplot(2, 3, 1)
    ax1.scatter(y_test_pred, test_residuals, alpha=0.5, s=20)
    ax1.axhline(y=0, color='r', linestyle='--', linewidth=2)
    ax1.set_xlabel('Predicted Price (£)')
    ax1.set_ylabel('Residuals (£)')
    ax1.set_title('Residuals vs Predicted (Test Set)')
    ax1.grid(True, alpha=0.3)
    
    # 2. Residuals vs Actual (Test Set)
    ax2 = plt.subplot(2, 3, 2)
    ax2.scatter(y_test, test_residuals, alpha=0.5, s=20)
    ax2.axhline(y=0, color='r', linestyle='--', linewidth=2)
    ax2.set_xlabel('Actual Price (£)')
    ax2.set_ylabel('Residuals (£)')
    ax2.set_title('Residuals vs Actual (Test Set)')
    ax2.grid(True, alpha=0.3)
    
    # 3. Histogram of Residuals
    ax3 = plt.subplot(2, 3, 3)
    ax3.hist(test_residuals, bins=50, edgecolor='black', alpha=0.7)
    ax3.axvline(x=0, color='r', linestyle='--', linewidth=2)
    ax3.set_xlabel('Residuals (£)')
    ax3.set_ylabel('Frequency')
    ax3.set_title('Distribution of Residuals (Test Set)')
    ax3.grid(True, alpha=0.3)
    
    # 4. Q-Q Plot
    ax4 = plt.subplot(2, 3, 4)
    stats.probplot(test_residuals, dist="norm", plot=ax4)
    ax4.set_title('Q-Q Plot (Normality Check)')
    ax4.grid(True, alpha=0.3)
    
    # 5. Scale-Location Plot (Sqrt of standardized residuals)
    ax5 = plt.subplot(2, 3, 5)
    standardized_residuals = test_residuals / np.std(test_residuals)
    ax5.scatter(y_test_pred, np.sqrt(np.abs(standardized_residuals)), alpha=0.5, s=20)
    ax5.set_xlabel('Predicted Price (£)')
    ax5.set_ylabel('√|Standardized Residuals|')
    ax5.set_title('Scale-Location Plot (Homoscedasticity Check)')
    ax5.grid(True, alpha=0.3)
    
    # 6. Predicted vs Actual
    ax6 = plt.subplot(2, 3, 6)
    ax6.scatter(y_test, y_test_pred, alpha=0.5, s=20)
    min_val = min(y_test.min(), y_test_pred.min())
    max_val = max(y_test.max(), y_test_pred.max())
    ax6.plot([min_val, max_val], [min_val, max_val], 'r--', linewidth=2, label='Perfect Prediction')
    ax6.set_xlabel('Actual Price (£)')
    ax6.set_ylabel('Predicted Price (£)')
    ax6.set_title('Predicted vs Actual (Test Set)')
    ax6.legend()
    ax6.grid(True, alpha=0.3)
    
    plt.tight_layout()
    if save_plots:
        plt.savefig('residual_analysis.png', dpi=300, bbox_inches='tight')
        print("\n✓ Residual plots saved as 'residual_analysis.png'")
    plt.show()
    
    # ========================================================================
    # 8. TRAIN VS TEST PERFORMANCE COMPARISON
    # ========================================================================
    print("\n" + "="*80)
    print("8. TRAIN VS TEST PERFORMANCE COMPARISON")
    print("="*80)
    
    print("\n{:<20} {:<15} {:<15} {:<15}".format("Metric", "Training", "Test", "Ratio (Test/Train)"))
    print("-" * 65)
    print("{:<20} {:<15.4f} {:<15.4f} {:<15.2f}".format(
        "R²", train_r2, test_r2, test_r2/train_r2 if train_r2 != 0 else 0))
    print("{:<20} £{:<14,.0f} £{:<14,.0f} {:<15.2f}".format(
        "RMSE", train_rmse, test_rmse, test_rmse/train_rmse))
    print("{:<20} £{:<14,.0f} £{:<14,.0f} {:<15.2f}".format(
        "MAE", train_mae, test_mae, test_mae/train_mae))
    
    # Overfitting assessment
    rmse_ratio = test_rmse / train_rmse
    if rmse_ratio > 1.5:
        print("\n⚠ SEVERE OVERFITTING: Test RMSE is >1.5x training RMSE")
        print("   Recommendations: Regularization, feature reduction, more data")
    elif rmse_ratio > 1.2:
        print("\n⚠ MODERATE OVERFITTING: Test RMSE is >1.2x training RMSE")
        print("   Recommendations: Consider regularization or cross-validation")
    else:
        print("\n✓ GOOD GENERALIZATION: Test performance close to training")
    
    # ========================================================================
    # 9. BUSINESS/DOMAIN METRICS
    # ========================================================================
    print("\n" + "="*80)
    print("9. BUSINESS/DOMAIN METRICS")
    print("="*80)
    
    # Predictions within ±10%
    within_10pct = np.abs((y_test - y_test_pred) / y_test) <= 0.10
    pct_within_10 = np.mean(within_10pct) * 100
    
    # Predictions within ±20%
    within_20pct = np.abs((y_test - y_test_pred) / y_test) <= 0.20
    pct_within_20 = np.mean(within_20pct) * 100
    
    # Predictions within ±£50k
    within_50k = np.abs(y_test - y_test_pred) <= 50000
    pct_within_50k = np.mean(within_50k) * 100
    
    results['pct_within_10pct'] = pct_within_10
    results['pct_within_20pct'] = pct_within_20
    results['pct_within_50k'] = pct_within_50k
    
    print(f"\nPredictions within ±10%:    {pct_within_10:.1f}%")
    print(f"Predictions within ±20%:    {pct_within_20:.1f}%")
    print(f"Predictions within ±£50k:   {pct_within_50k:.1f}%")
    
    # Underpricing vs Overpricing
    underpriced = np.sum(y_test_pred < y_test)
    overpriced = np.sum(y_test_pred > y_test)
    
    print(f"\nUnderpriced (predicted < actual): {underpriced} ({underpriced/len(y_test)*100:.1f}%)")
    print(f"Overpriced (predicted > actual):  {overpriced} ({overpriced/len(y_test)*100:.1f}%)")
    
    # Error by price segment
    print("\n--- Error Analysis by Price Segment ---")
    
    price_segments = [
        (0, 200000, "Budget (<£200k)"),
        (200000, 400000, "Mid-range (£200k-£400k)"),
        (400000, 600000, "Premium (£400k-£600k)"),
        (600000, float('inf'), "Luxury (>£600k)")
    ]
    
    print("\n{:<25} {:<10} {:<15} {:<15}".format("Segment", "Count", "MAE", "MAPE"))
    print("-" * 65)
    
    for low, high, label in price_segments:
        mask = (y_test >= low) & (y_test < high)
        if np.sum(mask) > 0:
            segment_mae = mean_absolute_error(y_test[mask], y_test_pred[mask])
            segment_mape = np.mean(np.abs((y_test[mask] - y_test_pred[mask]) / y_test[mask])) * 100
            print("{:<25} {:<10} £{:<14,.0f} {:<15.2f}%".format(
                label, np.sum(mask), segment_mae, segment_mape))
    
    # ========================================================================
    # 10. PREDICTION INTERVALS (95% Confidence)
    # ========================================================================
    print("\n" + "="*80)
    print("10. PREDICTION INTERVALS")
    print("="*80)
    
    # Calculate prediction standard error
    residual_std = np.std(test_residuals)
    
    # 95% prediction interval (±1.96 * std)
    lower_bound = y_test_pred - 1.96 * residual_std
    upper_bound = y_test_pred + 1.96 * residual_std
    
    # Check coverage
    coverage = np.mean((y_test >= lower_bound) & (y_test <= upper_bound)) * 100
    
    print(f"\n95% Prediction Interval Width: ±£{1.96 * residual_std:,.0f}")
    print(f"Actual Coverage: {coverage:.1f}% (should be ~95%)")
    
    if coverage < 90:
        print("⚠ Coverage below 90% - intervals may be too narrow")
    elif coverage > 98:
        print("⚠ Coverage above 98% - intervals may be too wide")
    else:
        print("✓ Good coverage - intervals appropriately calibrated")
    
    # Example predictions with intervals
    print("\n--- Example Predictions with 95% Intervals ---")
    sample_indices = np.random.choice(len(y_test), min(5, len(y_test)), replace=False)
    
    for idx in sample_indices:
        actual = y_test.iloc[idx] if isinstance(y_test, pd.Series) else y_test[idx]
        predicted = y_test_pred[idx]
        lower = lower_bound[idx]
        upper = upper_bound[idx]
        error = actual - predicted
        
        print(f"\nActual: £{actual:,.0f} | Predicted: £{predicted:,.0f} | Error: £{error:,.0f}")
        print(f"95% Interval: [£{lower:,.0f}, £{upper:,.0f}]")
    
    # ========================================================================
    # 11. FEATURE IMPORTANCE (if available)
    # ========================================================================
    print("\n" + "="*80)
    print("11. FEATURE IMPORTANCE")
    print("="*80)
    
    if hasattr(reg_model, 'coef_'):
        coefficients = reg_model.coef_
        
        # Get feature names if available
        if hasattr(X_train, 'columns'):
            feature_names = X_train.columns
        else:
            feature_names = [f'Feature_{i}' for i in range(X_train.shape[1])]
        
        # Create dataframe of coefficients
        coef_df = pd.DataFrame({
            'Feature': feature_names,
            'Coefficient': coefficients,
            'Abs_Coefficient': np.abs(coefficients)
        }).sort_values('Abs_Coefficient', ascending=False)
        
        print("\nTop 10 Most Important Features (by absolute coefficient):")
        print(coef_df.head(10).to_string(index=False))
        
        # Plot top 10 features
        plt.figure(figsize=(10, 6))
        top_features = coef_df.head(10)
        colors = ['green' if x > 0 else 'red' for x in top_features['Coefficient']]
        plt.barh(top_features['Feature'], top_features['Coefficient'], color=colors, alpha=0.7)
        plt.xlabel('Coefficient Value')
        plt.title('Top 10 Feature Coefficients (Green=Positive, Red=Negative)')
        plt.tight_layout()
        if save_plots:
            plt.savefig('feature_importance.png', dpi=300, bbox_inches='tight')
            print("\n✓ Feature importance plot saved as 'feature_importance.png'")
        plt.show()
    else:
        print("\nFeature coefficients not available for this model type")
    
    # ========================================================================
    # FINAL SUMMARY
    # ========================================================================
    print("\n" + "="*80)
    print("EVALUATION SUMMARY")
    print("="*80)
    
    print(f"\n{'='*80}")
    print("KEY METRICS (Test Set)")
    print(f"{'='*80}")
    print(f"R² Score:                  {test_r2:.4f}")
    print(f"RMSE:                      £{test_rmse:,.0f}")
    print(f"MAE:                       £{test_mae:,.0f}")
    print(f"Within ±10%:               {pct_within_10:.1f}%")
    print(f"Cross-Validated RMSE:      £{cv_rmse_mean:,.0f} (±£{cv_rmse_std:,.0f})")
    
    print(f"\n{'='*80}")
    print("MODEL ASSESSMENT")
    print(f"{'='*80}")
    
    # Overall assessment
    if test_r2 > 0.8 and test_rmse / train_rmse < 1.2:
        print("✓ EXCELLENT: Strong predictive power with good generalization")
    elif test_r2 > 0.6 and test_rmse / train_rmse < 1.3:
        print("✓ GOOD: Decent predictive power, acceptable generalization")
    elif test_r2 > 0.4:
        print("⚠ FAIR: Moderate predictive power, consider improvements")
    else:
        print("✗ POOR: Weak predictive power, significant improvements needed")
    
    print("\n" + "="*80)
    print("EVALUATION COMPLETE")
    print("="*80)
    
    return results


# ============================================================================
# USAGE EXAMPLE
# ============================================================================

if __name__ == "__main__":
    # This is an example of how to use the evaluation function
    # In practice, you would have already fitted your model and split your data
    
    from sklearn.linear_model import LinearRegression, Ridge, Lasso
    from sklearn.model_selection import train_test_split
    from sklearn.preprocessing import StandardScaler
    
    print("\n" + "="*80)
    print("EXAMPLE USAGE")
    print("="*80)
    print("\nGenerating sample housing data for demonstration...")
    
    # Generate sample data
    np.random.seed(42)
    n_samples = 1000
    n_features = 10
    
    X = np.random.randn(n_samples, n_features)
    y = (50000 + 
         X[:, 0] * 30000 +  # Square meters effect
         X[:, 1] * 20000 +  # Bedrooms effect
         X[:, 2] * 15000 +  # Location effect
         np.random.randn(n_samples) * 20000)  # Noise
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    
    # Scale features
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)
    
    # Fit model
    print("\nFitting Linear Regression model...")
    reg_model = LinearRegression()
    reg_model.fit(X_train, y_train)
    
    print("\nRunning comprehensive evaluation...\n")
    
    # Run evaluation
    results = evaluate_regression_model(
        reg_model, X_train, y_train, X_test, y_test,
        cv_folds=5, save_plots=True
    )
    
    print("\n\nTo use with your own model, simply call:")
    print("results = evaluate_regression_model(reg_model, X_train, y_train, X_test, y_test)")