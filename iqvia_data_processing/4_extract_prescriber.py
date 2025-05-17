import pandas as pd
import time 
import os
from multiprocessing import Pool, cpu_count
from tqdm import tqdm

def prscrbr_count(patient_data):
    # Step 1: Get unique pat_ids
    patient_id, most_recent_date, processed_patient_ids = patient_data
    
    # Skip already processed patients
    if patient_id in processed_patient_ids:
        print("Patient ID:", patient_id, "already processed, next!", flush = True)
        return None
    
    patient_df = prscbr_df[prscbr_df['pat_id'] == patient_id]
    # Step 2: Initialize a list of dates
    start_date_180_days = most_recent_date - pd.Timedelta(days=180)
    start_date_2_years = most_recent_date - pd.Timedelta(days=2*365)

    prscbr_last_2_years = patient_df[(patient_df['most_recent_date'] > start_date_2_years) & 
                                  (patient_df['most_recent_date'] <= most_recent_date)]['prscbr_id'].nunique()
    prscrbr_last_180_days = patient_df[(patient_df['most_recent_date'] > start_date_180_days) & 
                                  (patient_df['most_recent_date'] <= most_recent_date)]['prscbr_id'].nunique()

    return {
        'pat_id': patient_id,
        'most_recent_date': most_recent_date,
        'prscbr_last_2_years': float(prscbr_last_2_years),
        'prscrbr_last_180_days': float(prscrbr_last_180_days),
    }

if __name__ == "__main__":
    start_time = time.time()
    id_fields = ['pat_id','most_recent_date']
    id_df = pd.read_csv('/sharefolder/wanglab/MME/MME_features_final.csv', usecols = id_fields)

    # Ensure most_recent_date is in datetime format
    id_df['most_recent_date'] = pd.to_datetime(id_df['most_recent_date'], errors='coerce')
    # Initialize the columns for the three features
    
    prscbr_fields = ['pat_id','to_dt','prscbr_id']
    prscbr_df = pd.read_csv('/sharefolder/wanglab/MME/mme_data_final_combined.csv', usecols = prscbr_fields)
    prscbr_df.rename(columns={'to_dt': 'most_recent_date'}, inplace=True)
    prscbr_df['most_recent_date'] = pd.to_datetime(prscbr_df['most_recent_date'], errors='coerce')

    prscbr_df['prscbr_last_2_years'] = 0
    prscbr_df['prscbr_last_180_days'] = 0

    # Merge dataframes
    merged_df = pd.merge(prscbr_df, id_df, on=['pat_id','most_recent_date'], how='left')
    merged_df = merged_df.groupby('pat_id')['most_recent_date'].max().reset_index()

    # Load existing data if the file already exists
    output_file_path = '/sharefolder/wanglab/MME/prscbr_features.csv'
    processed_patient_ids = []
    if os.path.exists(output_file_path):
        existing_data = pd.read_csv(output_file_path)
        processed_patient_ids = existing_data['pat_id'].tolist()

    patient_data = [(row['pat_id'], pd.to_datetime(row['most_recent_date']), processed_patient_ids) for _, row in merged_df.iterrows()]

    # Use multiprocessing Pool
    with Pool(processes=cpu_count()) as pool:
        results = list(tqdm(pool.imap(prscrbr_count, patient_data), total=len(patient_data)))

    results = [result for result in results if result is not None]
    results = pd.Dataframe(results)
    if os.path.exists(output_file_path):
        existing_data = pd.read_csv(output_file_path)
        results = pd.concat([existing_data, results], ignore_index=True)
    results.to_csv(output_file_path, index=False)

    print(f"Final data saved successfully to {output_file_path}")
    end_time = time.time()
    print(f"Elapsed Time: {end_time - start_time:.2f} seconds")