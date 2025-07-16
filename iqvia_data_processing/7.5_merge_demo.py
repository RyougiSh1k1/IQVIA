"""
Final OUD Patient Feature Extraction Pipeline
=============================================

This script combines multiple steps to extract and merge demographic features
for OUD patients into one cohesive pipeline. It includes:

1. Extracting age, gender, and ZIP3 from enrollment synthetic files
2. Extracting pay_type from enroll2 files
3. Merging all demographic features with ZIP-level aggregated census data
4. Saving a clean final dataset ready for ML modeling

Output:
- final_OUD_ML_dataset.csv
"""

import pandas as pd
import os
from multiprocessing import Pool, cpu_count
from tqdm import tqdm
import concurrent.futures

# Global paths to data directories and input files
ENROLL_SYNTH_DIR = '/sharefolder/IQVIA/enroll_synth/csv_in_parts'
ENROLL2_BASE = '/sharefolder/IQVIA/enroll2_'
PATIENT_CSV = '/sharefolder/wanglab/MME/final_ML_dataset.csv'
ZIP_CSV = 'uszips.csv'


def calculate_age(yob, year):
    """
    Calculate age from year of birth (yob) and reference year.
    Returns None if inputs are invalid.
    """
    try:
        return int(year) - int(float(yob))
    except:
        return None


def extract_demographics(patient_list):
    """
    Read enrollment synthetic data and extract demographics (sex, yob, zip3)
    for OUD patients in the provided patient list.

    Uses multiprocessing to read and filter files efficiently.
    """
    header_data = ['der_sex', 'der_yob', 'pat_id', 'pat_region', 'pat_state',
                   'pat_zip3', 'grp_indv_cd', 'mh_cd', 'enr_rel']
    csv_files = [os.path.join(ENROLL_SYNTH_DIR, f) for f in os.listdir(ENROLL_SYNTH_DIR) if f.endswith('.csv')]
    args_list = [(f, header_data, set(patient_list)) for f in csv_files]

    def read_and_filter(args):
        file_path, headers, pat_ids = args
        df = pd.read_csv(file_path, sep='|', header=None, dtype=str)
        df.columns = headers
        return df[df['pat_id'].isin(pat_ids)]

    print(f"Reading enrollment synth data using {min(4, cpu_count())} processes...")
    with Pool(processes=min(4, cpu_count())) as pool:
        all_parts = list(tqdm(pool.imap(read_and_filter, args_list), total=len(args_list), desc="Processing Enroll Synth"))

    demographics_df = pd.concat(all_parts, ignore_index=True)
    return demographics_df


def extract_payment_types(all_data):
    """
    Extract payment type for each patient from enroll2_{year} files.
    Matches (pat_id, month_id) based on the most recent date in each patient's record.
    """
    # Add index date (yyyymm) and year columns
    all_data['index_date'] = pd.to_datetime(all_data['most_recent_date']).dt.strftime('%Y%m')
    all_data['index_year'] = all_data['index_date'].str[:4]

    # Create mapping: year -> list of (pat_id, month_id)
    grouped = all_data.groupby('index_year')[['pat_id', 'index_date']].apply(
        lambda x: list(x.itertuples(index=False, name=None))
    ).to_dict()

    def read_enroll(year_pat_list):
        """
        Read enroll2 file for a given year and filter rows by (pat_id, month_id)
        """
        year, pat_date_list = year_pat_list
        folder = os.path.join(ENROLL2_BASE + str(year), 'csv_in_parts')
        if not os.path.exists(folder):
            return pd.DataFrame()

        header = ['pat_id', 'mstr_enroll_cd', 'prd_type', 'pay_type', 'pcob_type', 'mcob_type', 'month_id']
        pat_set = set(pat_date_list)
        all_data = []

        for file in os.listdir(folder):
            if not file.endswith('.csv'):
                continue
            df = pd.read_csv(os.path.join(folder, file), sep='|', header=None, dtype=str)
            df.columns = header
            df = df[df[['pat_id', 'month_id']].apply(tuple, axis=1).isin(pat_set)]
            all_data.append(df)

        return pd.concat(all_data, ignore_index=True) if all_data else pd.DataFrame(columns=header)

    print("Reading enrollment payment data in parallel...")
    with concurrent.futures.ProcessPoolExecutor() as executor:
        results = list(executor.map(read_enroll, grouped.items()))

    return pd.concat(results, ignore_index=True)


def aggregate_zip_data():
    """
    Load ZIP-level census data and aggregate by ZIP3 (first 3 digits).
    Returns a DataFrame with mean values for all numeric SDOH columns.
    """
    zip_df = pd.read_csv(ZIP_CSV)
    zip_df['zip'] = zip_df['zip'].astype(str).str.zfill(5)
    zip_df['zip3'] = zip_df['zip'].str[:3]
    aggregated = zip_df.groupby('zip3').mean(numeric_only=True).reset_index()
    return aggregated


def main():
    """
    Main function to run the demographic pipeline end-to-end.
    Includes:
    - Demographic extraction
    - Payment type merging
    - ZIP-level feature integration
    - Final dataset export
    """
    print("Loading main patient list...")
    all_patients = pd.read_csv(PATIENT_CSV)
    all_patients['most_recent_date'] = pd.to_datetime(all_patients['most_recent_date'])
    all_patients['year'] = all_patients['most_recent_date'].dt.year

    # Take the first observed year per patient
    base_patients = all_patients.sort_values(by=['pat_id', 'year']).drop_duplicates('pat_id')

    print("Extracting demographic info (age, gender, ZIP3)...")
    demo_df = extract_demographics(base_patients['pat_id'].tolist())

    # Merge extracted demographics with baseline patient data
    merged = pd.merge(demo_df, base_patients, on='pat_id', how='inner')
    merged['age'] = merged.apply(lambda x: calculate_age(x['der_yob'], x['year']), axis=1)

    # Filter to adult patients (18–65)
    #merged = merged[(merged['age'] >= 18) & (merged['age'] =< 65)]
    print(f"Extracted {len(merged)} adult patient records.")
    merged = merged[['pat_id', 'age', 'der_sex', 'pat_zip3']]

    print("Extracting payment type info...")
    pay_df = extract_payment_types(all_patients)
    pay_df = pay_df[['pat_id', 'pay_type']].drop_duplicates()

    print("Aggregating ZIP-level data...")
    zip_df = aggregate_zip_data()

    print("Merging all data sources...")
    merged_all = all_patients.merge(merged, on='pat_id', how='left')
    merged_all = merged_all.merge(pay_df, on='pat_id', how='left')

    # Format ZIP3 for joining
    merged_all['pat_zip3'] = merged_all['pat_zip3'].astype(str).str.zfill(3)
    zip_df['zip3'] = zip_df['zip3'].astype(str).str.zfill(3)
    merged_all = merged_all.merge(zip_df, left_on='pat_zip3', right_on='zip3', how='left')

    # Drop helper and redundant columns
    merged_all = merged_all.drop(columns=['zip3'])
    merged_all = merged_all.dropna()  # Remove rows with any missing values
    merged_all = merged_all[(merged_all['pay_type'] != 'U') & (merged_all['der_sex'] != 'U')]

    print(f"Saving final cleaned dataset with {len(merged_all)} rows...")
    merged_all.to_csv('final_OUD_ML_dataset.csv', index=False)
    print("Pipeline complete! Dataset saved as 'final_OUD_ML_dataset.csv'")


if __name__ == "__main__":
    main()