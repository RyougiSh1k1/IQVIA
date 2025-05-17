'''
This file produces the MME equivalent for each prescription by merges on the OPIOID_FINDER.CSV file. Also removes common incorrections.
'''
import pandas as pd
import os
import time

years = [str(year) for year in range(2006, 2023)]
for year in years:
    # Read the CSV files
    start_time = time.time()
    df = pd.read_csv(f'/sharefolder/wanglab/iqvia_ndc_{year}.csv', dtype={'daw': 'str'})

    # Remove duplicate rows
    df = df.drop_duplicates()

    # Filter data
    filtered_data = df[df['from_dt'] == df['to_dt']]

    # Process opioid_finder data
    opioid_finder = pd.read_csv('/home/stofer@chapman.edu/iqvia_data_processing/OPIOID_FINDER.CSV')
    col_to_drop = ['START', 'STOP', 'I', 'SCD', 'DENOMINATOR_VALUE', 'DENOMINATOR_UNITS',
                'ACTIVE_MOIETY_NAME', 'SCHEDULE', 'RXNAV_STR', 'STRENGTH']
    opioid_finder = opioid_finder.drop(col_to_drop, axis=1)
    opioid_finder.columns = [col.lower() for col in opioid_finder.columns]

    # Remove duplicate NDC codes
    opioid_finder = opioid_finder.drop_duplicates(subset='ndc')

    # Merge data and save
    merged_data = pd.merge(filtered_data, opioid_finder, on='ndc', how='left')
    merged_data.to_csv(f'/sharefolder/wanglab/MME/mme_data_all_{year}.csv', index=False)

    # Read existing NDCs from the file if it exists
    txt_file_path = '/sharefolder/wanglab/ndc_not_found.txt'
    existing_ndcs = set()
    try:
        with open(txt_file_path, 'r') as file:
            for line in file:
                existing_ndcs.add(line.strip())
    except FileNotFoundError:
        pass

    # Find all NDC codes not found and combine with existing
    ndc_left_only = filtered_data[~filtered_data['ndc'].isin(opioid_finder['ndc'])]['ndc'].unique()
    all_ndcs = existing_ndcs.union(set(ndc_left_only))

    # Write unique NDCs back to the file
    with open(txt_file_path, 'w') as file:
        for ndc in all_ndcs:  # No sorting
            file.write(f"{ndc}\n")

    # Read the merged data and filter
    #mme_df = pd.read_csv('/sharefolder/wanglab/mme_data_all_2006.csv')
    with open(txt_file_path, 'r') as file:
        codes_to_remove = [line.strip() for line in file]

    mme_df = merged_data

    # Remove rows where 'ndc' is in the list of codes to remove
    mme_df = mme_df[~mme_df['ndc'].isin(codes_to_remove)]

    # Additional filtering
    mme_df = mme_df[(mme_df['dayssup'] >= 1) & (mme_df['dayssup'] < 999)]
    mme_df = mme_df[mme_df['dayssup'] % 1 == 0]
    mme_df = mme_df[mme_df['numerator_value'] > 0]

    # Calculate MME
    mme_df['MME'] = (mme_df['numerator_value'] / mme_df['dayssup']) * mme_df['mme_conversion_factor']

    # Check statistics
    print(f'{year} stastistics: Num_Value, Days Sup, MME_Conv_Factor, MME:')
    print(mme_df['numerator_value'].describe(), flush = True)
    print(mme_df['dayssup'].describe(), flush = True)
    print(mme_df['mme_conversion_factor'].describe(), flush = True)
    print(mme_df['MME'].describe(), flush = True)

    # Remove duplicate rows in the final DataFrame
    mme_df = mme_df.drop_duplicates()

    # Save the final DataFrame
    mme_df.to_csv(f'/sharefolder/wanglab/MME/mme_data_final_{year}.csv', index=False)
    end_time = time.time()
    print(f"Done with {year}, next!")
    elapsed_time = end_time - start_time
    print(f"Elapsed Time: {elapsed_time:.2f} seconds", flush = True)
