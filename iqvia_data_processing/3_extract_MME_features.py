'''
This file uses all MME recorded observations from 2006-2022 and extracts features for ORS algorithm
'''

import pandas as pd
import os
import time
from multiprocessing import Pool, cpu_count
from tqdm import tqdm

# Function to process each patient's data
def process_patient(patient_data):
    patient_id, most_recent_date, processed_patient_ids = patient_data

    
    # Skip already processed patients
    if patient_id in processed_patient_ids:
        print("Patient ID:", patient_id, "already processed, next!", flush = True)
        return None
    
    print("Patient ID:", patient_id, flush = True)
    # Define date ranges
    start_date_365_days = most_recent_date - pd.Timedelta(days=365)
    start_date_2_years = most_recent_date - pd.Timedelta(days=2*365)
    
    # Filter original dataframe for the current patient
    patient_df = mme_df[mme_df['pat_id'] == patient_id]
    
    # Calculate total MMEs for the last 365 days
    mme_last_365_days = patient_df[(patient_df['to_dt'] > start_date_365_days) & 
                                   (patient_df['to_dt'] <= most_recent_date)]['MME'].sum()
    
    # Calculate total MMEs for the last 2 years
    mme_last_2_years = patient_df[(patient_df['to_dt'] > start_date_2_years) & 
                                  (patient_df['to_dt'] <= most_recent_date)]['MME'].sum()
    
    # Calculate total MMEs for more than 1 year prior to most recent date
    mme_prior_1_year = patient_df[patient_df['to_dt'] < start_date_365_days]['MME'].sum()
    
    # Calculate the number of opioid prescriptions with daily MMEs > 120 in the last 2 years
    mme_120_2_years = patient_df[(patient_df['to_dt'] > start_date_2_years) & (patient_df['to_dt'] <= most_recent_date) & (patient_df['daily_MME'] > 120)].shape[0]

    return {
        'pat_id': patient_id,
        'most_recent_date': most_recent_date,
        'MME_last_365_days': float(mme_last_365_days),
        'MME_last_2_years': float(mme_last_2_years),
        'MME_prior_1_year': float(mme_prior_1_year),
        'MME_120_2_years': float(mme_120_2_years)  # Add this new feature
    }

if __name__ == "__main__":
    start_time = time.time()
    cols = ['pat_id','to_dt', 'MME', 'daily_MME']
    mme_df = pd.read_csv('/sharefolder/wanglab/MME/mme_data_final_combined.csv', usecols = cols)
    mme_df = mme_df[mme_df['daily_MME'] <= 200]

    # Ensure to_dt is in datetime format
    mme_df['to_dt'] = pd.to_datetime(mme_df['to_dt'], errors='coerce')
    most_recent_df = mme_df.groupby('pat_id')['to_dt'].max().reset_index()
    most_recent_df.rename(columns={'to_dt': 'most_recent_date'}, inplace=True)
    
    # Initialize the columns for the three features
    most_recent_df['MME_last_365_days'] = 0
    most_recent_df['MME_last_2_years'] = 0
    most_recent_df['MME_prior_1_year'] = 0
    most_recent_df['MME_120_2_years'] = 0


    # Load existing data if the file already exists
    output_file_path = '/sharefolder/wanglab/MME/MME_features_200_max.csv'
    processed_patient_ids = []
    if os.path.exists(output_file_path):
        existing_data = pd.read_csv(output_file_path)
        processed_patient_ids = existing_data['pat_id'].tolist()

    # Prepare patient data for processing
    patient_data = [(row['pat_id'], pd.to_datetime(row['most_recent_date']), processed_patient_ids) for _, row in most_recent_df.iterrows()]
    
    # Use multiprocessing Pool
    with Pool(processes=cpu_count()) as pool:
        results = list(tqdm(pool.imap(process_patient, patient_data), total=len(patient_data)))

    # Filter out None results
    results = [result for result in results if result is not None]

    # Save results
    if results:
        final_results = pd.DataFrame(results)
        if os.path.exists(output_file_path):
            existing_data = pd.read_csv(output_file_path)
            final_results = pd.concat([existing_data, final_results], ignore_index=True)
        final_results.to_csv(output_file_path, index=False)

    print(f"Final data saved successfully to {output_file_path}")
    end_time = time.time()
    print(f"Elapsed Time: {end_time - start_time:.2f} seconds")