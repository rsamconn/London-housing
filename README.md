# London-housing
Applying some simple ML models while learning about housing data in London.

## Running the code
The code is contained in notebooks which can each be run independently:
1. `predicting_prices.ipynb` - Applying simple linear regression models to predict house prices.
2. `in-depth/EDA_LONG.ipynb` - Detailed exploratory data analysis of the housing data.
3. `in-depth/Regression_anlaysis_LONG.ipynb` - Detailed investigation using linear regression models to predict house prices.

## To do
1. ~Choose a dateset from [Kaggle](https://www.kaggle.com/datasets?search=london+hous)~
2. ~EDA - completed in the `start_EDA` branch, (merged back to main) in `EDA.ipynb`~
3. ~Initial ML analysis (test some KNNs for fun) - not much success but completed in `Classification_analysis.ipynb`~
4. ~Main ML analysis: find the best linear regression model and optimize it - completed in the `start_ML` branch (merged back to main) in `Regression_anlaysis.ipynb`.~
5. ~Create a shortened version of the ML notebook for public display.~
6. Create a shortened version of the EDA notebook for public display.

## Goals
Learn about housing data:
- Best price predictors
- Average price by no. of bedrooms by borough
- Price change over time

ML models to use:
- scikit-learn supervised ML models
- Classification: KNN, logistic
- Regression (linear, Lasso, Ridge)
- Try feature selection with Lasso regression
- Can also try tree-based models for regression and classification