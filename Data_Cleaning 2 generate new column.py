import random
import numpy as np
import pandas as pd
import pickle
import time
import matplotlib.pyplot as plt

# Step01: Connecting 7 OHLCV CSV files into the project
file_path = 'D:/OneDrive - The Pennsylvania State University/01 DAAN 545/00 Project/00 Data Source/Combined_Data_DS1_DS2.csv'

# Step02: Read and show the CSV file with encoding='iso-8859-1'
df_bg = pd.read_csv(file_path, encoding='iso-8859-1')
# Filter the data based on the condition
df_bg = df_bg[df_bg['Number_of_use_Printing_Cylinder'] == 3]
print("Test Read Dataset")
print(df_bg)
print('\n' + '-' * 50 + '\n')  # Add a separator between columns

# Create new columns to store the minimum values
df_bg['New_Printing_Gap_1'] = float('nan')
df_bg['New_Printing_Gap_2'] = float('nan')
df_bg['New_Printing_Gap_3'] = float('nan')

# Iterate over each row in the DataFrame
for index, row in df_bg.iterrows():
    try:
        # Get the values from each column and exclude non-numeric values
        values = row[['Printing_Cylinder_Gap_1', 'Printing_Cylinder_Gap_2', 'Printing_Cylinder_Gap_3', 'Printing_Cylinder_Gap_4']]
        values = pd.to_numeric(values, errors='coerce')
        values = values.dropna()

        # Get the minimum values and sort them in ascending order
        min_values = values.nsmallest(3).sort_values()

        # Store the minimum values in the corresponding columns
        df_bg.at[index, 'New_Printing_Gap_1'] = min_values.iloc[0] if len(min_values) > 0 else np.nan
        df_bg.at[index, 'New_Printing_Gap_2'] = min_values.iloc[1] if len(min_values) > 1 else np.nan
        df_bg.at[index, 'New_Printing_Gap_3'] = min_values.iloc[2] if len(min_values) > 2 else np.nan
    except TypeError:
        # Terminate the row and skip to the next row
        print(f"Terminating row {index} due to TypeError")
        continue



# Calculate the average value
df_bg['New_Average_Printing_Gap'] = df_bg[['New_Printing_Gap_1', 'New_Printing_Gap_2', 'New_Printing_Gap_3']].mean(axis=1)

# Print the updated DataFrame
print(df_bg)

# Specify the full path where you want to save the Excel file
file_path = 'D:/OneDrive - The Pennsylvania State University/01 DAAN 545/00 Project/00 Data Source/Sorting_Printing_Gap/Data_with_3_Printing_Gap.xlsx'

# Export the DataFrame to the specified path
df_bg.to_excel(file_path, index=False)
