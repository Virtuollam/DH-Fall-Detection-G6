import pandas as pd

# Replace these with your actual CSV file paths
file1 = 'standing_20240205_154806.csv'
file2 = 'standing_20240205_154935.csv'
file3 = 'standing_20240205_155041.csv'

# Read the CSV files
df1 = pd.read_csv(file1)
df2 = pd.read_csv(file2)
df3 = pd.read_csv(file3)

# Combine the dataframes
combined_df = pd.concat([df1, df2, df3])

# Assuming 'date_column_name' is the name of your date or timestamp column
# Replace 'date_column_name' with the actual column name
combined_df.sort_values('date_column_name', inplace=True)

# Reset index if you want a clean, sequential index in the final CSV
combined_df.reset_index(drop=True, inplace=True)

# Save the combined dataframe to a new CSV file
combined_df.to_csv('standinglong.csv', index=False)
