import pandas as pd


# Step01: Connecting 7 OHLCV CSV files into the project
file_path = 'D:/OneDrive - The Siam Cement Public Company Limited/07 Python Projects/PSU_DAAN545_Group_Project/.venv/01_DAAN_545_Dataset1.csv'
file_path2 = 'D:/OneDrive - The Siam Cement Public Company Limited/07 Python Projects/PSU_DAAN545_Group_Project/.venv/02_DAAN_545_Dataset2.csv'

df = pd.read_csv(file_path, encoding='iso-8859-1')
df2 = pd.read_csv(file_path2, encoding='iso-8859-1')
print('Merged data from the 2 CSV files')
# Step02: Append 2 data sets together name as appended_df
appended_df = pd.concat([df, df2], axis=0, ignore_index=True)
print(appended_df)

print('\n' + '-'*50 + '\n')  # Add a separator between columns
print('Show how many missing values each column ?')
# Step03: Show how many missing values each column ?
missing_values = appended_df.isnull().sum()
rows_with_nulls = appended_df[appended_df.isnull().any(axis=1)] # Row number with null data
print(missing_values)
print('\n' + '-'*50 + '\n')  # Add a separator between columns

print('Filter rows where [Order_Number] is not null and [BCT1] is null because this data can not be use for box strength analytics')
#Step04 Filter rows where 'Order_Number' is not null and 'BCT1' is null
condition = appended_df['Order_Number'].notnull() & appended_df['BCT1'].isnull()
# Remove rows that meet the condition
appended_df = appended_df[~condition]

print(appended_df)


#Step05 Create a mapping dictionary for Raw_Mat_Strength_Additive to make new a new column as a percentage of each Raw_Mat_Strength_Additive
strength_mapping = {
    'G0': '0%',
    'G1': '5%',
    'G2': '10%',
    'G3': '15%',
    'G4': '20%',
    'G5': '25%',
    'G6': '30%'
}

appended_df['Raw_Mat_Strength_Additive_Percent'] = appended_df['Raw_Mat_Strength_Additive'].map(strength_mapping)




#Step06 Replace null values in multiple columns
columns_to_fill = ['GL','GLWeigth(Grammage)', 'BM', 'BMWeigth(Grammage)', 'BL', 'BLWeigth(Grammage)', 'CM', 'CMWeigth(Grammage)', 'CL', 'CLWeigth(Grammage)']
appended_df[columns_to_fill] = appended_df[columns_to_fill].fillna('No Use')


print('\n' + '-'*50 + '\n')  # Add a separator between columns


#Step07 Filter rows where Finished_Product_Wid and Finished_Product_Leg and Finished_Product_Hig is null
condition2 = appended_df['Finished_Product_Wid'].isnull() | appended_df['Finished_Product_Leg'].isnull() | appended_df['Finished_Product_Hig'].isnull()
# Remove rows that meet the condition
appended_df = appended_df[~condition2]



print('Show how many missing values each column ?')
# Step08: Show how many missing values each column ?
missing_values = appended_df.isnull().sum()
rows_with_nulls = appended_df[appended_df.isnull().any(axis=1)] # Row number with null data
print(missing_values)

print('\n' + '-'*50 + '\n')  # Add a separator between columns

print(appended_df)

# Step09: Specify the location and filename for the Excel file
excel_file_path = 'D:/OneDrive - The Pennsylvania State University/01 DAAN 545/00 Project/00 Data Source/Combined_Data_DS1_DS2.xlsx'

# Step10: Export the DataFrame to Excel
appended_df.to_excel(excel_file_path, index=False)