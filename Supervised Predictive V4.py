import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from statsmodels.iolib.summary2 import summary_col
from statsmodels.tools.tools import add_constant
from statsmodels.discrete.discrete_model import Logit
from sklearn.preprocessing import StandardScaler
from imblearn.over_sampling import SMOTENC
from sklearn.metrics import accuracy_score
from itertools import combinations

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
related_vars = ["ECT_Avg", "Feed_Roll_Gap", "New_Printing_Gap_1", "New_Printing_Gap_2", "New_Printing_Gap_3", "Speed_Act"]

# Generate dummy variables for categorical features
dummies = pd.get_dummies(df[['Raw_Mat_Strength_Additive', 'Raw_Mat_Strength_Type']].astype(str), drop_first=True)

# Check for highly predictive variables
correlation_matrix = df[related_vars].corr()
highly_predictive_vars = [var for var in related_vars if abs(correlation_matrix[var].loc[correlation_matrix[var].index != var].max()) == 1]

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
    accuracy_scores = []
    model_combinations = []

    # Generate all possible combinations of related variables
    for r in range(1, len(related_vars) + 1):
        for combination in combinations(related_vars, r):
            model_combinations.append(combination)

    for combination in model_combinations:
        X_ = pd.concat([dummies, df[list(combination)]], axis=1).astype(float)
        X_ = add_constant(X_)
        # Align indices of X_ with y_res
        X_ = X_.iloc[:len(y_res)].reset_index(drop=True)
        y_res = y_res.iloc[:len(X_)].reset_index(drop=True)
        # Verify alignment
        print(f"Indices of X_ for combination {combination}:", X_.index)
        print(f"Indices of y_res for combination {combination}:", y_res.index)
        result = Logit(y_res, X_).fit_regularized(maxiter=2000, method='l1', alpha=0.1)
        main_results.append(result)

        # Predictions
        y_pred = result.predict(X_)
        y_pred = (y_pred >= 0.5).astype(int)  # Convert predicted probabilities to binary predictions

        # Evaluate the model
        accuracy = accuracy_score(y_res, y_pred)
        accuracy_scores.append(accuracy)

    # Sort the accuracy scores and model combinations in descending order
    accuracy_scores, model_combinations = zip(*sorted(zip(accuracy_scores, model_combinations), reverse=True))

    # Create model names for the x-axis
    model_names = [f"Model {i + 1}" for i in range(len(model_combinations))]

    # Plot line chart to compare the accuracy of each model
    plt.figure(figsize=(12, 8))
    plt.plot(model_names, accuracy_scores, marker='o')
    plt.xlabel('Model')
    plt.ylabel('Accuracy Score')
    plt.title('Accuracy of Each Model')
    plt.xticks(rotation=90)
    plt.grid(True)
    plt.tight_layout()  # Adjust layout to accommodate rotated x-axis labels
    plt.show()

    # Print the accuracy scores of each model
    for i, accuracy in enumerate(accuracy_scores):
        print(f"Accuracy of Model {i + 1} (Combination {model_combinations[i]}): {accuracy}")

    # Print the best model's accuracy score and model number
    best_accuracy = accuracy_scores[0]
    best_model_combination = model_combinations[0]
    print(f"The best model is Model 1 with an accuracy score of {best_accuracy} using combination {best_model_combination}")

    # Generate prediction equation for each model
    prediction_equations = []
    for i, result in enumerate(main_results):
        equation = f"Prediction Equation for Model {i + 1} (Combination {model_combinations[i]}): "
        for j, var in enumerate(result.params.index):
            if var == "const":
                equation += f"{result.params[j]:.4f} + "
            else:
                equation += f"{result.params[j]:.4f} * {var} + "
        equation = equation[:-3]  # Remove the last "+"
        prediction_equations.append(equation)

    # Generate summary statistics
    summary_col1 = summary_col(main_results, stars=True, regressor_order=related_vars)
    print(summary_col1)

    # Specify the path for saving the Excel file
    excel_path = 'D:/OneDrive - The Pennsylvania State University/01 DAAN 545/00 Project/00 Data Source/Week3_Raw_Data/Sorting_Printing_Gap/model_summary.xlsx'

    # Create an Excel file with three worksheets
    with pd.ExcelWriter(excel_path) as writer:
        accuracy_df = pd.DataFrame({'Model': model_names, 'Model Number': range(1, len(model_names) + 1), 'Accuracy Score': accuracy_scores})
        accuracy_df.to_excel(writer, sheet_name='Accuracy Scores', index=False)
        pd.DataFrame(list(prediction_equations), columns=['Prediction Equation']).to_excel(writer, sheet_name='Prediction Equations', index=False)
        summary_col1.tables[0].to_excel(writer, sheet_name='Summary Statistics', index=False, header=False)