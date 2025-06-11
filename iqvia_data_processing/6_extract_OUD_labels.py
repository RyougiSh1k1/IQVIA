"""
This file extracts ICD labels from IQVIA claims files to classify patients with Opioid Use Disorder (OUD) events.
Creates binary labels: 1 for patients with OUD events, 0 for patients without OUD events.

The script looks for ICD codes related to opioid use disorders in the actual claims data
and creates target labels for the machine learning model.
"""

import pandas as pd
import time
import os
import re
from multiprocessing import Pool, cpu_count
from tqdm import tqdm

def load_oud_icd_codes(csv_path='extracted_icd_codes.csv'):
    """
    Load OUD-related ICD codes from the provided CSV file
    Returns a list of regex patterns to match ICD codes with wildcards
    """
    print("Loading OUD ICD codes...")
    
    # Read the ICD codes CSV
    icd_df = pd.read_csv(csv_path)
    
    # Extract unique ICD codes
    icd_codes = icd_df['ICD_Code'].unique()
    
    # Convert ICD codes to regex patterns (replace X/x with \d for any digit)
    icd_patterns = []
    for code in icd_codes:
        # Replace X or x with \d (any digit) and escape other special characters
        pattern = code.replace('X', r'\d').replace('x', r'\d')
        pattern = '^' + re.escape(pattern).replace(r'\\d', r'\d') + '$'
        icd_patterns.append(pattern)
    
    print(f"Loaded {len(icd_patterns)} ICD code patterns")
    return icd_patterns, list(icd_codes)

def read_claims_header(year='2006'):
    """
    Read the header for claims files
    """
    header_file = f'/sharefolder/IQVIA/header/header_claims_{year}.csv'
    with open(header_file, 'r') as f:
        header = f.readline().strip().split('|')
    return header

def check_icd_match(icd_code, icd_patterns):
    """
    Check if an ICD code matches any of the OUD patterns
    """
    if pd.isna(icd_code) or icd_code == '':
        return False
    
    icd_code = str(icd_code).strip()
    
    for pattern in icd_patterns:
        if re.match(pattern, icd_code):
            return True
    
    return False

def process_claims_file(args):
    """
    Process a single claims CSV file to extract OUD labels
    """
    file_path, header, icd_patterns, file_num, total_files = args
    
    print(f"Processing file {file_num}/{total_files}: {os.path.basename(file_path)}", flush=True)
    
    try:
        # Read the CSV file
        df = pd.read_csv(file_path, sep='|', header=None, dtype=str, low_memory=False)
        df.columns = header
        
        # ICD columns based on data dictionary
        icd_columns = ['diag1', 'diag2', 'diag3', 'diag4', 'diag5']
        
        # Initialize results list
        results = []
        
        # Process each row
        for idx, row in df.iterrows():
            pat_id = row['pat_id']
            
            # Check each diagnosis column
            matched_codes = []
            for col in icd_columns:
                if col in df.columns and not pd.isna(row[col]):
                    icd_code = str(row[col]).strip()
                    if icd_code and check_icd_match(icd_code, icd_patterns):
                        matched_codes.append(icd_code)
            
            # If any matches found, record the patient
            if matched_codes:
                results.append({
                    'pat_id': pat_id,
                    'matched_icd_codes': ','.join(matched_codes),
                    'oud_label': 1,
                    'service_date': row.get('from_dt', '')  # Using from_dt as service date
                })
        
        print(f"File {file_num}: Found {len(results)} patients with OUD codes", flush=True)
        return results
        
    except Exception as e:
        print(f"Error processing file {file_path}: {str(e)}", flush=True)
        return []

def extract_oud_labels_year(year='2006'):
    """
    Extract OUD labels for a specific year
    """
    print(f"\nProcessing year {year}...")
    
    # Load ICD patterns
    icd_patterns, original_codes = load_oud_icd_codes('extracted_icd_codes.csv')
    
    # Get header
    header = read_claims_header(year)
    print(f"Header columns: {header[:10]}...")  # Show first 10 columns
    
    # Get all CSV files for the year
    csv_dir = f'/sharefolder/IQVIA/claims_{year}/csv_in_parts'
    csv_files = [os.path.join(csv_dir, f) for f in os.listdir(csv_dir) if f.endswith('.csv')]
    csv_files.sort()
    
    print(f"Found {len(csv_files)} CSV files to process")
    
    # Prepare arguments for multiprocessing
    args_list = [(f, header, icd_patterns, i+1, len(csv_files)) 
                 for i, f in enumerate(csv_files)]
    
    # Process files in parallel
    all_results = []
    with Pool(processes=min(cpu_count(), 8)) as pool:  # Limit to 8 processes
        results_list = list(tqdm(
            pool.imap(process_claims_file, args_list),
            total=len(csv_files),
            desc=f"Processing {year} claims files"
        ))
        
        # Flatten results
        for results in results_list:
            all_results.extend(results)
    
    print(f"\nTotal patients with OUD codes in {year}: {len(all_results)}")
    
    # Convert to DataFrame
    if all_results:
        oud_df = pd.DataFrame(all_results)
        
        # Remove duplicates (keep first occurrence)
        oud_df = oud_df.drop_duplicates(subset=['pat_id'], keep='first')
        print(f"Unique patients with OUD codes: {len(oud_df)}")
        
        # Save year-specific results
        output_path = f'/sharefolder/wanglab/MME/oud_patients_{year}.csv'
        oud_df.to_csv(output_path, index=False)
        print(f"Saved OUD patients for {year} to: {output_path}")
        
        return oud_df
    else:
        print(f"No OUD patients found in {year}")
        return pd.DataFrame()

def create_final_labels(year='2006'):
    """
    Create final dataset with OUD labels for all patients
    """
    print(f"\nCreating final labels for {year}...")
    
    # Load the feature dataset
    features_path = '/sharefolder/wanglab/MME/final_features.csv'
    if os.path.exists(features_path):
        features_df = pd.read_csv(features_path)
        print(f"Loaded {len(features_df)} patients from features file")
    else:
        print(f"Features file not found at {features_path}")
        # Try to use the MME data as a proxy for patient list
        mme_path = f'/sharefolder/wanglab/MME/mme_data_final_{year}.csv'
        if os.path.exists(mme_path):
            mme_df = pd.read_csv(mme_path, usecols=['pat_id'])
            features_df = mme_df.drop_duplicates()
            print(f"Using MME data - found {len(features_df)} unique patients")
        else:
            print("No patient list available")
            return None
    
    # Load OUD patients
    oud_path = f'/sharefolder/wanglab/MME/oud_patients_{year}.csv'
    if os.path.exists(oud_path):
        oud_df = pd.read_csv(oud_path)
        oud_patients = set(oud_df['pat_id'].astype(str))
        print(f"Loaded {len(oud_patients)} OUD patients")
    else:
        print("OUD patients file not found")
        oud_patients = set()
    
    # Create labels
    features_df['pat_id_str'] = features_df['pat_id'].astype(str)
    features_df['oud_label'] = features_df['pat_id_str'].apply(
        lambda x: 1 if x in oud_patients else 0
    )
    
    # Add ICD codes for OUD patients
    if len(oud_patients) > 0:
        oud_df['pat_id_str'] = oud_df['pat_id'].astype(str)
        features_df = features_df.merge(
            oud_df[['pat_id_str', 'matched_icd_codes', 'service_date']],
            on='pat_id_str',
            how='left'
        )
    else:
        features_df['matched_icd_codes'] = ''
        features_df['service_date'] = ''
    
    # Clean up
    features_df = features_df.drop('pat_id_str', axis=1)
    
    # Summary statistics
    print(f"\nLabel Summary for {year}:")
    print(f"Total patients: {len(features_df)}")
    print(f"Patients with OUD (label=1): {features_df['oud_label'].sum()}")
    print(f"Patients without OUD (label=0): {len(features_df) - features_df['oud_label'].sum()}")
    print(f"OUD prevalence: {features_df['oud_label'].mean():.4%}")
    
    # Save final labeled dataset
    output_path = f'/sharefolder/wanglab/MME/final_dataset_with_labels_{year}.csv'
    features_df.to_csv(output_path, index=False)
    print(f"\nFinal labeled dataset saved to: {output_path}")
    
    return features_df

def main():
    """
    Main function to extract OUD labels
    """
    start_time = time.time()
    
    print("="*80)
    print("OUD LABEL EXTRACTION FROM CLAIMS DATA")
    print("="*80)
    
    # Process year 2006 first as requested
    year = '2006'
    
    try:
        # Extract OUD patients from claims
        oud_df = extract_oud_labels_year(year)
        
        # Create final labeled dataset
        final_df = create_final_labels(year)
        
        if final_df is not None:
            # Display sample of OUD patients
            print("\nSample of patients with OUD:")
            oud_sample = final_df[final_df['oud_label'] == 1].head(10)
            if len(oud_sample) > 0:
                print(oud_sample[['pat_id', 'matched_icd_codes', 'oud_label']])
            else:
                print("No OUD patients found in sample")
        
    except Exception as e:
        print(f"Error during processing: {str(e)}")
        import traceback
        traceback.print_exc()
    
    end_time = time.time()
    elapsed_time = end_time - start_time
    print(f"\nTotal processing time: {elapsed_time/60:.2f} minutes")
    
    print("\nExtraction complete!")
    print("Next steps:")
    print("1. Review the oud_patients_2006.csv file for OUD cases")
    print("2. Check final_dataset_with_labels_2006.csv for the complete labeled dataset")
    print("3. Run for other years (2007-2022) as needed")

if __name__ == "__main__":
    main()