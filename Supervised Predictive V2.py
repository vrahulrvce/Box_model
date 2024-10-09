import os
import pandas as pd
import numpy as np
from statsmodels.regression.linear_model import OLS
from statsmodels.tools.tools import add_constant
from statsmodels.discrete.discrete_model import Logit
from sklearn.preprocessing import StandardScaler
from imblearn.over_sampling import SMOTENC
from statsmodels.iolib.summary2 import summary_col

# Step01: Connecting the CSV file into the project
file_path = 'D:/OneDrive - The Pennsylvania State University/01 DAAN 545/00 Project/00 Data Source/Week3_Raw_Data/Sorting_Printing_Gap/Data_with_3_Printing_Gap.csv'
df = pd.read_csv(file_path, encoding='iso-8859-1')

# Filter the data based on the condition
df = df[df['Number_of_use_Printing_Cylinder'] == 3]

# Create the 'Rejection_Result' column based on the condition
df['Rejection_Result'] = np.where(
    (df['BCT_Avg'] < df['BCT_Spec']) | (abs((df['BCT_Avg'] - df['BCT_Spec']) / df['BCT_Spec']) < 0.03),
    0, 1
)

# Specify the target variable (Y) and related variables (X)
y = df['Rejection_Result']
related_vars = ["ECT_Avg", "Feed_Roll_Gap", "New_Printing_Gap_1", "New_Printing_Gap_2", "New_Printing_Gap_3",
                "Speed_Act"]

# Generate dummy variables for categorical features
dummies = pd.get_dummies(df[['Raw_Mat_Strength_Additive', 'Raw_Mat_Strength_Type']].astype(str), drop_first=True)

# Check for highly predictive variables
correlation_matrix = df[related_vars].corr()
highly_predictive_vars = []
for var in related_vars:
    if abs(correlation_matrix[var].loc[correlation_matrix[var].index != var].max()) == 1:
        highly_predictive_vars.append(var)

# Remove highly predictive variables
related_vars = [var for var in related_vars if var not in highly_predictive_vars]
print("Variables after removing highly predictive ones:", related_vars)

# Standardize the data
scaler = StandardScaler()
df[related_vars] = scaler.fit_transform(df[related_vars])

# Check for missing or infinite values in the data
if df[related_vars].isnull().values.any() or np.isinf(df[related_vars]).values.any():
    print("Data contains missing or infinite values. Please clean the data before proceeding.")
else:
    # Balance the dataset using SMOTENC
    X = pd.concat([dummies, df[related_vars]], axis=1)
    categorical_features = [i for i, col in enumerate(X.columns) if 'Raw_Mat_Strength' in col]
    smote_nc = SMOTENC(categorical_features=categorical_features, random_state=42)
    X_res, y_res = smote_nc.fit_resample(X, y)

    # Reset indices to ensure alignment
    X_res = X_res.reset_index(drop=True)
    y_res = y_res.reset_index(drop=True)

    # Verify alignment
    print("Indices of X_res:", X_res.index)
    print("Indices of y_res:", y_res.index)

    # Logistic regression analysis with regularization
    main_results = []
    r_squared_values = []
    for var in related_vars:
        X_ = pd.concat([dummies, df[[var]]], axis=1).astype(float)
        X_ = add_constant(X_)
        # Align indices of X_ with y_res
        X_ = X_.iloc[:len(y_res)].reset_index(drop=True)
        y_res = y_res.iloc[:len(X_)].reset_index(drop=True)
        # Verify alignment
        print(f"Indices of X_ for variable {var}:", X_.index)
        print(f"Indices of y_res for variable {var}:", y_res.index)
        result = Logit(y_res, X_).fit_regularized(maxiter=2000, method='l1', alpha=0.1)
        main_results.append(result)

        # Calculate McFadden's pseudo R²
        llf = result.llf  # Log-likelihood of the fitted model
        llnull = result.llnull  # Log-likelihood of the null model
        pseudo_r2 = 1 - (llf / llnull)
        r_squared_values.append(pseudo_r2)

    # Generate prediction equation for each model
    for i, result in enumerate(main_results):
        equation = f"Prediction Equation for Model {i + 1}: "
        for j, var in enumerate(result.params.index):
            if var == "const":
                equation += f"{result.params[j]:.4f} + "
            else:
                equation += f"{result.params[j]:.4f} * {var} + "
        equation = equation[:-3]  # Remove the last "+"
        print(equation)
        print(f"R² for Model {i + 1}: {r_squared_values[i]:.4f}")

    # Generate summary statistics
    summary_col1 = summary_col(main_results, stars=True, regressor_order=related_vars)
    print(summary_col1)