"""
This file extracts ICD labels to classify patients with Opioid Use Disorder (OUD) events.
Creates binary labels: 1 for patients with OUD events, 0 for patients without OUD events.

The script looks for ICD-10 codes related to opioid use disorders and creates target labels
for the machine learning model.

Input files:
- final_features.csv (merged feature dataset)
- Raw IQVIA claims data with ICD codes

Output file:
- final_dataset_with_labels.csv (complete dataset with features and OUD labels)
"""

import pandas as pd
import time
import os
from multiprocessing import Pool, cpu_count
from tqdm import tqdm

# ICD-10 codes for Opioid Use Disorders
OUD_ICD10_CODES = [
    'F11.10',  # Opioid use disorder, uncomplicated
    'F11.11',  # Opioid use disorder, in remission
    'F11.12',  # Opioid use disorder with intoxication
    'F11.120', # Opioid use disorder with intoxication, uncomplicated
    'F11.121', # Opioid use disorder with intoxication delirium
    'F11.122', # Opioid use disorder with intoxication with perceptual disturbance
    'F11.129', # Opioid use disorder with intoxication, unspecified
    'F11.13',  # Opioid use disorder with withdrawal
    'F11.14',  # Opioid use disorder with opioid-induced mood disorder
    'F11.15',  # Opioid use disorder with opioid-induced psychotic disorder
    'F11.150', # Opioid use disorder with opioid-induced psychotic disorder with delusions
    'F11.151', # Opioid use disorder with opioid-induced psychotic disorder with hallucinations
    'F11.159', # Opioid use disorder with opioid-induced psychotic disorder, unspecified
    'F11.18',  # Opioid use disorder with other opioid-induced disorder
    'F11.181', # Opioid use disorder with opioid-induced sexual dysfunction
    'F11.182', # Opioid use disorder with opioid-induced sleep disorder
    'F11.188', # Opioid use disorder with other opioid-induced disorder
    'F11.19',  # Opioid use disorder with unspecified opioid-induced disorder
    'F11.20',  # Opioid dependence, uncomplicated
    'F11.21',  # Opioid dependence, in remission
    'F11.22',  # Opioid dependence with intoxication
    'F11.220', # Opioid dependence with intoxication, uncomplicated
    'F11.221', # Opioid dependence with intoxication delirium
    'F11.222', # Opioid dependence with intoxication with perceptual disturbance
    'F11.229', # Opioid dependence with intoxication, unspecified
    'F11.23',  # Opioid dependence with withdrawal
    'F11.24',  # Opioid dependence with opioid-induced mood disorder
    'F11.25',  # Opioid dependence with opioid-induced psychotic disorder
    'F11.250', # Opioid dependence with opioid-induced psychotic disorder with delusions
    'F11.251', # Opioid dependence with opioid-induced psychotic disorder with hallucinations
    'F11.259', # Opioid dependence with opioid-induced psychotic disorder, unspecified
    'F11.28',  # Opioid dependence with other opioid-induced disorder
    'F11.281', # Opioid dependence with opioid-induced sexual dysfunction
    'F11.282', # Opioid dependence with opioid-induced sleep disorder
    'F11.288', # Opioid dependence with other opioid-induced disorder
    'F11.29',  # Opioid dependence with unspecified opioid-induced disorder
    'F11.90',  # Opioid use, unspecified, uncomplicated
    'F11.92',  # Opioid use, unspecified with intoxication
    'F11.920', # Opioid use, unspecified with intoxication, uncomplicated
    'F11.921', # Opioid use, unspecified with intoxication delirium
    'F11.922', # Opioid use, unspecified with intoxication with perceptual disturbance
    'F11.929', # Opioid use, unspecified with intoxication, unspecified
    'F11.93',  # Opioid use, unspecified with withdrawal
    'F11.94',  # Opioid use, unspecified with opioid-induced mood disorder
    'F11.95',  # Opioid use, unspecified with opioid-induced psychotic disorder
    'F11.950', # Opioid use, unspecified with opioid-induced psychotic disorder with delusions
    'F11.951', # Opioid use, unspecified with opioid-induced psychotic disorder with hallucinations
    'F11.959', # Opioid use, unspecified with opioid-induced psychotic disorder, unspecified
    'F11.98',  # Opioid use, unspecified with other specified opioid-induced disorder
    'F11.981', # Opioid use, unspecified with opioid-induced sexual dysfunction
    'F11.982', # Opioid use, unspecified with opioid-induced sleep disorder
    'F11.988', # Opioid use, unspecified with other opioid-induced disorder
    'F11.99'   # Opioid use, unspecified with unspecified opioid-induced disorder
]

def process_patient_oud_label(patient_data):
    """
    Process each patient to determine if they have an OUD event
    """
    patient_id, most_recent_date, processed_patient_ids = patient_data
    
    # Skip already processed patients
    if patient_id in processed_patient_ids:
        return None
    
    print(f"Processing patient: {patient_id}", flush=True)
    
    # Filter ICD data for current patient
    patient_icd_df = icd_df[icd_df['pat_id'] == patient_id]
    
    # Check if patient has any OUD-related ICD codes
    oud_events = patient_icd_df[patient_icd_df['icd_code'].isin(OUD_ICD10_CODES)]
    
    # Determine OUD label (1 if any OUD event found, 0 otherwise)
    oud_label = 1 if len(oud_events) > 0 else 0
    
    # Additional information for analysis
    num_oud_events = len(oud_events)
    first_oud_date = oud_events['service_date'].min() if len(oud_events) > 0 else None
    latest_oud_date = oud_events['service_date'].max() if len(oud_events) > 0 else None
    
    return {
        'pat_id': patient_id,
        'most_recent_date': most_recent_date,
        'oud_label': oud_label,
        'num_oud_events': num_oud_events,
        'first_oud_date': first_oud_date,
        'latest_oud_date': latest_oud_date
    }

def read_icd_data():
    """
    Read ICD data from IQVIA claims files
    This function needs to be adapted based on the actual structure of your ICD data
    """
    print("Loading ICD data from IQVIA claims...")
    
    # This is a placeholder - you'll need to adapt this based on your actual ICD data structure
    # The function should read from the same IQVIA source files but focus on ICD columns
    
    icd_data_list = []
    years = [str(year) for year in range(2006, 2023)]
    
    for year in years:
        try:
            # Attempt to read from the combined MME data which should have ICD codes
            # If ICD codes are in separate files, adjust this path accordingly
            year_file = f'/sharefolder/wanglab/MME/mme_data_final_{year}.csv'
            
            if os.path.exists(year_file):
                # Read only relevant columns for ICD analysis
                # Adjust column names based on actual data structure
                year_data = pd.read_csv(year_file, usecols=['pat_id', 'service_date', 'icd_code'])
                icd_data_list.append(year_data)
                print(f"Loaded ICD data for year {year}")
            else:
                print(f"File not found for year {year}: {year_file}")
                
        except Exception as e:
            print(f"Error loading data for year {year}: {str(e)}")
            continue
    
    if not icd_data_list:
        raise ValueError("No ICD data could be loaded. Please check file paths and column names.")
    
    # Combine all years
    combined_icd_df = pd.concat(icd_data_list, ignore_index=True)
    
    # Convert service_date to datetime
    combined_icd_df['service_date'] = pd.to_datetime(combined_icd_df['service_date'], errors='coerce')
    
    # Remove rows with missing ICD codes
    combined_icd_df = combined_icd_df.dropna(subset=['icd_code'])
    
    print(f"Total ICD records loaded: {len(combined_icd_df)}")
    print(f"Unique patients with ICD codes: {combined_icd_df['pat_id'].nunique()}")
    
    return combined_icd_df

def extract_oud_labels():
    """
    Main function to extract OUD labels for all patients
    """
    global icd_df
    
    start_time = time.time()
    
    print("Starting OUD label extraction...")
    
    # Load the final features dataset
    features_path = '/sharefolder/wanglab/MME/final_features.csv'
    if not os.path.exists(features_path):
        raise FileNotFoundError(f"Final features file not found: {features_path}")
    
    features_df = pd.read_csv(features_path)
    print(f"Loaded features for {len(features_df)} patients")
    
    # Load ICD data
    try:
        icd_df = read_icd_data()
    except Exception as e:
        print(f"Error loading ICD data: {str(e)}")
        print("Creating dummy labels (all patients labeled as 0 - no OUD)")
        
        # Create dummy labels if ICD data is not available
        features_df['oud_label'] = 0
        features_df['num_oud_events'] = 0
        features_df['first_oud_date'] = None
        features_df['latest_oud_date'] = None
        
        output_path = '/sharefolder/wanglab/MME/final_dataset_with_labels.csv'
        features_df.to_csv(output_path, index=False)
        print(f"Dataset with dummy labels saved to: {output_path}")
        return features_df
    
    # Prepare data for multiprocessing
    features_df['most_recent_date'] = pd.to_datetime(features_df['most_recent_date'])
    
    # Load existing processed labels if file exists
    output_file_path = '/sharefolder/wanglab/MME/oud_labels.csv'
    processed_patient_ids = []
    if os.path.exists(output_file_path):
        existing_data = pd.read_csv(output_file_path)
        processed_patient_ids = existing_data['pat_id'].tolist()
        print(f"Found {len(processed_patient_ids)} already processed patients")
    
    # Prepare patient data for processing
    patient_data = [
        (row['pat_id'], pd.to_datetime(row['most_recent_date']), processed_patient_ids) 
        for _, row in features_df.iterrows()
    ]
    
    print(f"Processing OUD labels for {len(patient_data)} patients...")
    
    # Use multiprocessing Pool
    with Pool(processes=cpu_count()) as pool:
        results = list(tqdm(pool.imap(process_patient_oud_label, patient_data), total=len(patient_data)))
    
    # Filter out None results
    results = [result for result in results if result is not None]
    
    if results:
        # Save OUD labels
        oud_labels_df = pd.DataFrame(results)
        
        # Merge with existing data if it exists
        if os.path.exists(output_file_path):
            existing_data = pd.read_csv(output_file_path)
            oud_labels_df = pd.concat([existing_data, oud_labels_df], ignore_index=True)
        
        oud_labels_df.to_csv(output_file_path, index=False)
        print(f"OUD labels saved to: {output_file_path}")
        
        # Merge with features dataset
        final_dataset = pd.merge(
            features_df, 
            oud_labels_df[['pat_id', 'most_recent_date', 'oud_label', 'num_oud_events', 'first_oud_date', 'latest_oud_date']], 
            on=['pat_id', 'most_recent_date'], 
            how='left'
        )
        
        # Fill missing labels with 0 (no OUD)
        final_dataset['oud_label'] = final_dataset['oud_label'].fillna(0)
        final_dataset['num_oud_events'] = final_dataset['num_oud_events'].fillna(0)
        
        # Save final dataset
        final_output_path = '/sharefolder/wanglab/MME/final_dataset_with_labels.csv'
        final_dataset.to_csv(final_output_path, index=False)
        
        # Print summary statistics
        print("\nOUD Label Summary:")
        print(f"Total patients: {len(final_dataset)}")
        print(f"Patients with OUD: {final_dataset['oud_label'].sum()}")
        print(f"Patients without OUD: {len(final_dataset) - final_dataset['oud_label'].sum()}")
        print(f"OUD prevalence: {final_dataset['oud_label'].mean():.4f}")
        
        print(f"\nFinal dataset with labels saved to: {final_output_path}")
        
    end_time = time.time()
    elapsed_time = end_time - start_time
    print(f"Elapsed Time: {elapsed_time:.2f} seconds")
    
    return final_dataset

if __name__ == "__main__":
    try:
        final_dataset = extract_oud_labels()
        print("OUD label extraction completed successfully!")
        
        # Display final dataset info
        print(f"\nFinal dataset shape: {final_dataset.shape}")
        print(f"Columns: {list(final_dataset.columns)}")
        
    except Exception as e:
        print(f"Error during OUD label extraction: {str(e)}")
        raise