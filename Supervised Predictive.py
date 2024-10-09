import os
import pandas as pd
import numpy as np
from statsmodels.regression.linear_model import OLS
from statsmodels.tools.tools import add_constant
from statsmodels.discrete.discrete_model import Logit
from statsmodels.iolib.summary2 import summary_col
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
from sklearn.cluster import DBSCAN, KMeans
from sklearn.cluster import MeanShift
from sklearn.datasets import make_blobs
import seaborn as sns
from sklearn.cluster import AgglomerativeClustering
from sklearn.metrics import silhouette_score
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error
import joblib
from imblearn.over_sampling import SMOTENC

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
related_vars = ["BCT_Avg", "ECT_Avg", "Feed_Roll_Gap", "New_Printing_Gap_1", "New_Printing_Gap_2", "New_Printing_Gap_3", "Speed_Act"]

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
    # Logistic regression analysis with regularization
    main_results = []
    for var in related_vars:
        X_ = pd.concat([dummies, df[[var]]], axis=1).astype(float)
        X_ = add_constant(X_)
        result = Logit(y, X_).fit_regularized(maxiter=1000, method='l1', alpha=0.5)
        main_results.append(result)

    # Fit the logistic regression model with all related variables
    X_ = pd.concat([dummies, df[related_vars]], axis=1).astype(float)
    X_ = add_constant(X_)
    result = Logit(y, X_).fit_regularized(maxiter=1000, method='l1', alpha=0.5)
    main_results.append(result)

    # Generate the summary statistics
    summary_col1 = summary_col(main_results, stars=True, regressor_order=related_vars)
    print(summary_col1)

    # Compare with sklearn using regularization
    X_ = pd.concat([dummies, df[related_vars]], axis=1)
    X_ = add_constant(X_)
    model = LogisticRegression(max_iter=1000, solver='liblinear', penalty='l1', C=1.0)
    result = model.fit(X_, y)
    coef = pd.Series(result.coef_[0], index=X_.columns)
    print(coef)

    # Check the average marginal effect of sklearn model
    from sklearn.inspection import partial_dependence
    tes = partial_dependence(result, X_, features=['BCT_Avg'])

    # Average partial effect
    ape_results = []
    for result in main_results:
        ape_results.append(result.get_margeff(at='mean', method='dydx', atexog=None, dummy=False, count=False))
        ape_results1 = ape_results[0].summary()
        print(ape_results1)

    # Additional logistic regression analysis
    additional_results = []
    for vars in [related_vars]:
        X_ = pd.concat([dummies, df[vars]], axis=1).astype(float)
        X_ = add_constant(X_)
        result = Logit(y, X_).fit_regularized(maxiter=1000, method='l1', alpha=0.5)
        additional_results.append(result)

    summary_col2 = summary_col(additional_results, stars=True, regressor_order=related_vars)
    print(summary_col2)

    # One issue is that "Rejection_Result" is very imbalanced.
    print(y.mean())

    # Let's try to balance it using SMOTE.
    for i in range(10):
        smote = SMOTENC(categorical_features=[dummies.columns.get_loc('Raw_Mat_Strength_Additive')])
        X = df.drop(columns=['Rejection_Result'])
        X_res, y_res = smote.fit_resample(X, y)
        dummies_res = pd.get_dummies(X_res[['Raw_Mat_Strength_Additive', 'Raw_Mat_Strength_Type']].astype(str), drop_first=True)
        fit_args = {'maxiter': 100, 'cov_type': 'cluster', 'cov_kwds': {'groups': X_res['gvkey_num']}}
        main_results_smote = []
        for var in related_vars:
            X_ = pd.concat([dummies_res, X_res[[var]]], axis=1).astype(float)
            X_ = add_constant(X_)
            result = Logit(y_res, X_).fit_regularized(disp=0, method='l1', alpha=0.5, **fit_args)
            main_results_smote.append(result)
        X_ = pd.concat([dummies_res, X_res[related_vars]], axis=1).astype(float)
        X_ = add_constant(X_)
        result = Logit(y_res, X_).fit_regularized(disp=0, method='l1', alpha=0.5, **fit_args)
        main_results_smote.append(result)
        print(summary_col(main_results_smote, stars=True, regressor_order=related_vars))