import random
import numpy as np
import pandas as pd
import pickle
import time
import matplotlib.pyplot as plt
import time
from mlxtend.frequent_patterns import apriori, association_rules, fpgrowth
import math
import seaborn as sns

# Connecting 7 OHLCV CSV files into the project
file_path = 'D:/OneDrive - The Pennsylvania State University/01 DAAN 545/00 Project/00 Data Source/Week3_Raw_Data/Sorting_Printing_Gap/Data_with_3_Printing_Gap.csv'

#Read and show the CSV file with encoding='iso-8859-1'
df_bg = pd.read_csv(file_path, encoding='iso-8859-1')

# Apply pd.get_dummies to a specific column (e.g., 'Column_Name')


# Define corresponding values
# Definir valores correspondientes
values = [
    'Gap Range: 0.0-1.0',
    'Gap Range: 1.0-2.0',
    'Gap Range: 2.0-3.0',
    'Gap Range: >3'
]

# Convertir valores a cadenas
values = [str(value) for value in values]

columns = ['New_Printing_Gap_1', 'New_Printing_Gap_2', 'New_Printing_Gap_3', 'New_Average_Printing_Gap',
           'Feed_Roll_Gap']
new_columns = ['Printing_Gap_1_Range', 'Printing_Gap_2_Range', 'Printing_Gap_3_Range', 'Avg_Printing_Gap_Range',
               'Feed_Roll_Gap_Range']

for col, new_col in zip(columns, new_columns):
    conditions = [
        (df_bg[col] <= 1.0),
        (df_bg[col] > 1.0) & (df_bg[col] <= 2.0),
        (df_bg[col] > 2.0) & (df_bg[col] <= 3.0),
        (df_bg[col] > 3.0)
    ]
    df_bg[new_col] = np.select(conditions, values, default='Unknown')

df_bg['Rejection_Result'] = np.where(
    (df_bg['BCT_Avg'] < df_bg['BCT_Spec']) | (abs((df_bg['BCT_Avg'] - df_bg['BCT_Spec']) / df_bg['BCT_Spec']) < 0.03),
    0, 1)
df_bg['Suitable_Hig'] = np.where((df_bg['Finished_Product_Hig'] < 600) & (df_bg['Finished_Product_Hig'] > 300), 1, 0)

df_bg['ECT_Result'] = np.where(
    (df_bg['ECT_Avg'] < 1.98),
    0, 1)

df_bg = df_bg[df_bg['Rejection_Result'] == 1]


def remove_outliers(df, column):
    Q1 = df[column].quantile(0.25)
    Q3 = df[column].quantile(0.75)
    IQR = Q3 - Q1
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR
    return df[(df[column] >= lower_bound) & (df[column] <= upper_bound)]


filtered_df1 = df_bg[((df_bg['Feed_Roll_Gap'] >= 1.0) & (df_bg['Feed_Roll_Gap'] <= 2.0) & df_bg[
    'New_Average_Printing_Gap'] >= 1.0) & (df_bg['New_Average_Printing_Gap'] <= 2.0) &
                     (df_bg['Raw_Mat_Strength_Type'] == 'C') &
                     (df_bg['Raw_Mat_Strength_Additive'] == 'G3') &
                     (df_bg['SKU'] == '1397-123-00G3')]

filtered_df1 = remove_outliers(filtered_df1, 'BCT_Avg')
filtered_df1 = remove_outliers(filtered_df1, 'New_Average_Printing_Gap')

print(filtered_df1)

sns.regplot(x=filtered_df1['BCT_Avg'], y=filtered_df1['New_Average_Printing_Gap'], ci=None)
plt.xlabel('BCT_Avg')
plt.ylabel('New_Average_Printing_Gap')
plt.title('Scatter Plot with Trend Line: BCT_Avg vs New_Average_Printing_Gap')
plt.show()

# Specify the columns you want to transform into dummy variables
columns_to_transform = ['Raw_Mat_Strength_Type', 'Raw_Mat_Strength_Additive', 'Avg_Printing_Gap_Range',
                        'Printing_Gap_1_Range', 'Printing_Gap_2_Range', 'Printing_Gap_3_Range', 'Feed_Roll_Gap_Range']

# Apply pd.get_dummies to the specified columns and set dtype=int
df_bg = pd.get_dummies(df_bg, columns=columns_to_transform, prefix=columns_to_transform, dtype=int)

print(df_bg)
print(df_bg.columns.tolist())

selected_columns = ['Raw_Mat_Strength_Type_B', 'Raw_Mat_Strength_Type_BC', 'Raw_Mat_Strength_Type_C',
                    'Raw_Mat_Strength_Additive_G0', 'Raw_Mat_Strength_Additive_G1', 'Raw_Mat_Strength_Additive_G2',
                    'Raw_Mat_Strength_Additive_G3', 'Raw_Mat_Strength_Additive_G4', 'Raw_Mat_Strength_Additive_G5',
                    'Raw_Mat_Strength_Additive_G6', 'Suitable_Hig', 'Avg_Printing_Gap_Range_Gap Range: 0.0-1.0',
                    'Avg_Printing_Gap_Range_Gap Range: 1.0-2.0', 'Avg_Printing_Gap_Range_Gap Range: 2.0-3.0',
                    'Avg_Printing_Gap_Range_Gap Range: >3', 'Printing_Gap_1_Range_Gap Range: 0.0-1.0',
                    'Printing_Gap_2_Range_Gap Range: 0.0-1.0', 'Printing_Gap_2_Range_Gap Range: 1.0-2.0',
                    'Printing_Gap_2_Range_Gap Range: 2.0-3.0', 'Printing_Gap_2_Range_Gap Range: >3',
                    'Printing_Gap_3_Range_Gap Range: 0.0-1.0', 'Printing_Gap_3_Range_Gap Range: 1.0-2.0',
                    'Printing_Gap_3_Range_Gap Range: 2.0-3.0', 'Printing_Gap_3_Range_Gap Range: >3',
                    'Feed_Roll_Gap_Range_Gap Range: 0.0-1.0', 'Feed_Roll_Gap_Range_Gap Range: 1.0-2.0',
                    'Feed_Roll_Gap_Range_Gap Range: 2.0-3.0', 'Feed_Roll_Gap_Range_Gap Range: >3','ECT_Result']

df_selected = df_bg[selected_columns]

print(df_selected)
# Sort the frequent itemsets by support in descending order
df_selected.to_excel(
    'D:/OneDrive - The Pennsylvania State University/01 DAAN 545/00 Project/00 Data Source/Week3_Raw_Data/Sorting_Printing_Gap/Binary_Data.xlsx',
    index=False)

#Use apriori function to view pattern of data and set min support at 0.10 then plot a graph
print('Apriori and sorting desc')
start_time_apriori = time.time()
freq_items = apriori(df_selected, min_support=0.05, use_colnames=True, verbose=1)
end_time_apriori = time.time()
execution_time_apriori = end_time_apriori - start_time_apriori

# Sort the frequent itemsets by support in descending order
sorted_itemsets = freq_items.sort_values(by='support', ascending=False)
print(sorted_itemsets.sort_values(by='support', ascending=False))
sorted_itemsets.to_excel(
    'D:/OneDrive - The Pennsylvania State University/01 DAAN 545/00 Project/00 Data Source/Week3_Raw_Data/Sorting_Printing_Gap/01_apriori_sorted_itemsets.xlsx',
    index=False)

# Export the sorted DataFrame to an Excel file


# Plot the support values of the frequent itemsets
plt.figure(figsize=(10, 6))
bars = plt.bar(sorted_itemsets.index.astype(str), sorted_itemsets['support'], color='skyblue')
plt.xlabel('Itemsets')
plt.ylabel('Support')
plt.title('Support of Frequent Itemsets')
plt.xticks(rotation=90)
# Add data labels to the bars
for i, bar in enumerate(bars):
    yval = bar.get_height()
    plt.text(bar.get_x() + bar.get_width() / 2, yval, round(yval, 2), ha='center', va='bottom')
    plt.text(bar.get_x() + bar.get_width() / 2, -0.05, sorted_itemsets['itemsets'].iloc[i], ha='center', va='top',
             rotation=90)

plt.show()

#Use fpgrowth function
print('FP-Growth and sorting desc')
start_time_fpgrowth = time.time()
frequent_itemsets = fpgrowth(df_selected, min_support=0.05, use_colnames=True, verbose=1)
end_time_fpgrowth = time.time()
execution_time_fpgrowth = end_time_fpgrowth - start_time_fpgrowth

# Sort the frequent itemsets by support in descending order
sorted_itemsets = frequent_itemsets.sort_values(by='support', ascending=False)
sorted_itemsets.to_excel(
    'D:/OneDrive - The Pennsylvania State University/01 DAAN 545/00 Project/00 Data Source/Week3_Raw_Data/Sorting_Printing_Gap/02_fpgrowth_sorted_itemsets.xlsx',
    index=False)

# Plot the support values of the frequent itemsets
plt.figure(figsize=(10, 6))
bars = plt.bar(sorted_itemsets.index.astype(str), sorted_itemsets['support'], color='skyblue')
plt.xlabel('Itemsets')
plt.ylabel('Support')
plt.title('Support of Frequent Itemsets (FP-Growth)')
plt.xticks(rotation=90)

# Add data labels to the bars
for i, bar in enumerate(bars):
    yval = bar.get_height()
    plt.text(bar.get_x() + bar.get_width() / 2, yval, round(yval, 2), ha='center', va='bottom')
    plt.text(bar.get_x() + bar.get_width() / 2, -0.05, sorted_itemsets['itemsets'].iloc[i], ha='center', va='top',
             rotation=90)

plt.show()
