'''
Combine all MME years together to one big file for later
'''

import pandas as pd
import time
import os

# List of years as strings
years = [str(year) for year in range(2006, 2023)]

# Initialize an empty list to store data frames
df_list = []

# Loop over each year, read the CSV, and append to the list
for year in years:
    start_time = time.time()
    
    # Read the CSV file for the current year
    mme_df_part = pd.read_csv(f'/sharefolder/wanglab/MME/mme_data_final_{year}.csv')
    
    # Append the data frame to the list
    df_list.append(mme_df_part)
    
    end_time = time.time()
    elapsed_time = end_time - start_time
    print(f"Elapsed Time for {year}: {elapsed_time:.2f} seconds", flush=True)

# Concatenate all data frames in the list into one big data frame
mme_df_combined = pd.concat(df_list, ignore_index=True)

# Save the combined data frame to a single CSV file
mme_df_combined.to_csv('/sharefolder/wanglab/MME/mme_data_final_combined.csv', index=False)

print("All years combined and saved successfully!")
