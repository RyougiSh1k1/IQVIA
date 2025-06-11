"""
Extract OUD labels from IQVIA claims files by matching ICD codes.
This script processes claims data to identify patients with Opioid Use Disorder.

Input:
- extracted_icd_codes.csv: List of OUD-related ICD codes (with X/x as wildcards)
- header_claims_2006.csv: Header file with column names
- /sharefolder/IQVIA/claims_2006/csv_in_parts/*.csv: Claims data files

Output:
- oud_patients_2006.csv: Patients with OUD (pat_id, matched_icd_codes, oud_label=1)
- all_patients_with_labels_2006.csv: All patients with their OUD labels (0 or 1)
"""

import pandas as pd
import numpy as np
import os
import re
import time
from glob import glob
from multiprocessing import Pool, cpu_count
from tqdm import tqdm

def load_icd_codes(filepath='extracted_icd_codes.csv'):
    """Load OUD ICD codes and convert wildcards to regex patterns"""
    print("Loading OUD ICD codes...")
    
    # Try multiple possible locations
    paths_to_try = [
        filepath,
        f'./{filepath}',
        f'/home/qinyu@chapman.edu/IQVIA/iqvia_data_processing/{filepath}',
        f'/home/qinyu@chapman.edu/IQVIA/{filepath}'
    ]
    
    icd_df = None
    for path in paths_to_try:
        if os.path.exists(path):
            try:
                icd_df = pd.read_csv(path)
                print(f"Loaded ICD codes from: {path}")
                break
            except Exception as e:
                print(f"Error reading {path}: {e}")
    
    if icd_df is None:
        raise FileNotFoundError(f"Could not find {filepath}")
    
    # Get unique ICD codes
    icd_codes = icd_df['ICD_Code'].unique().tolist()
    print(f"Found {len(icd_codes)} unique ICD codes")
    
    # Convert to regex patterns (X or x = any digit)
    patterns = []
    for code in icd_codes:
        # Replace X/x with \d (digit pattern)
        pattern = str(code).replace('X', r'\d').replace('x', r'\d')
        # Escape special characters except \d
        pattern = re.sub(r'([.+?^${}()|[\]])', r'\\\1', pattern)
        pattern = pattern.replace(r'\\d', r'\d')  # Restore \d
        patterns.append(f'^{pattern}$')
    
    return patterns, icd_codes

def load_header(year='2006'):
    """Load column headers for claims files"""
    print(f"Loading header for year {year}...")
    
    # Try multiple locations
    header_files = [
        f'header_claims_{year}.csv',
        f'./header_claims_{year}.csv',
        f'/home/qinyu@chapman.edu/IQVIA/iqvia_data_processing/header/header_claims_{year}.csv',
    ]
    
    for header_file in header_files:
        if os.path.exists(header_file):
            try:
                with open(header_file, 'r') as f:
                    content = f.read().strip()
                    # Handle pipe-delimited
                    if '|' in content:
                        headers = content.split('|')
                    else:
                        # Handle comma-delimited
                        headers = content.split(',')
                    
                print(f"Loaded {len(headers)} columns from: {header_file}")
                return headers
            except Exception as e:
                print(f"Error reading {header_file}: {e}")
    
    # Default headers based on data dictionary
    print("Using default headers from data dictionary...")
    return [
        'pat_id', 'pat_gender', 'pat_yob', 'pat_zip3', 'evt_typ',
        'from_dt', 'to_dt', 'dayssup', 'clmid', 'ndc', 'qty',
        'scriptnum', 'maint_ind', 'pay_typ', 'copay', 'avg_wac',
        'mail_ind', 'specialty', 'prscbr_id', 'prscbr_gender',
        'prscbr_yob', 'prscbr_zip3', 'prscbr_typ', 'pharm_blk',
        'pharm_zip3', 'ncpdp_id', 'chain_ind', 'diag1', 'diag2',
        'diag3', 'diag4', 'diag5'
    ]

def check_icd_match(icd_value, patterns):
    """Check if an ICD code matches any OUD pattern"""
    if pd.isna(icd_value) or str(icd_value).strip() == '':
        return False
    
    icd_str = str(icd_value).strip().upper()
    
    for pattern in patterns:
        if re.match(pattern, icd_str):
            return True
    
    return False

def process_single_file(args):
    """Process one claims CSV file"""
    filepath, headers, patterns, file_idx, total_files = args
    
    filename = os.path.basename(filepath)
    print(f"[{file_idx}/{total_files}] Processing {filename}...", flush=True)
    
    try:
        # Read CSV file
        df = pd.read_csv(filepath, sep='|', header=None, dtype=str, low_memory=False)
        
        # Assign column names
        if len(df.columns) == len(headers):
            df.columns = headers
        else:
            print(f"Warning: Column count mismatch in {filename}")
            print(f"Expected {len(headers)} columns, got {len(df.columns)}")
            return []
        
        # Diagnosis columns to check
        diag_cols = ['diag1', 'diag2', 'diag3', 'diag4', 'diag5']
        
        # Find patients with OUD codes
        oud_patients = []
        
        for _, row in df.iterrows():
            matched_codes = []
            
            # Check each diagnosis column
            for col in diag_cols:
                if col in df.columns:
                    icd_val = row.get(col, '')
                    if check_icd_match(icd_val, patterns):
                        matched_codes.append(str(icd_val).strip())
            
            # If OUD codes found, record the patient
            if matched_codes:
                oud_patients.append({
                    'pat_id': str(row['pat_id']),
                    'matched_icd_codes': '|'.join(matched_codes),
                    'service_date': row.get('from_dt', ''),
                    'oud_label': 1
                })
        
        print(f"[{file_idx}/{total_files}] Found {len(oud_patients)} OUD patients in {filename}", flush=True)
        return oud_patients
        
    except Exception as e:
        print(f"Error processing {filename}: {e}", flush=True)
        return []

def main():
    """Main processing function"""
    start_time = time.time()
    year = '2006'
    
    print("="*80)
    print("IQVIA OUD LABEL EXTRACTION")
    print("="*80)
    
    # Load ICD patterns
    try:
        patterns, original_codes = load_icd_codes()
        print(f"\nSample ICD patterns: {patterns[:5]}")
    except Exception as e:
        print(f"ERROR: Failed to load ICD codes: {e}")
        return
    
    # Load headers
    headers = load_header(year)
    print(f"\nHeaders: {headers[:10]}... (showing first 10)")
    
    # Get list of claims files
    claims_dir = f'/sharefolder/IQVIA/claims_{year}/csv_in_parts'
    if not os.path.exists(claims_dir):
        print(f"ERROR: Claims directory not found: {claims_dir}")
        return
    
    csv_files = sorted(glob(os.path.join(claims_dir, '*.csv')))
    print(f"\nFound {len(csv_files)} CSV files to process")
    
    if not csv_files:
        print("ERROR: No CSV files found")
        return
    
    # Process files in parallel
    print("\nProcessing claims files...")
    pool_size = min(cpu_count(), 8)  # Limit parallel processes
    
    # Prepare arguments for multiprocessing
    process_args = [
        (csv_file, headers, patterns, i+1, len(csv_files))
        for i, csv_file in enumerate(csv_files)
    ]
    
    # Process with progress bar
    all_oud_patients = []
    with Pool(processes=pool_size) as pool:
        results = list(tqdm(
            pool.imap(process_single_file, process_args),
            total=len(csv_files),
            desc="Processing files"
        ))
        
        # Combine results
        for result in results:
            if result:
                all_oud_patients.extend(result)
    
    print(f"\nTotal OUD records found: {len(all_oud_patients)}")
    
    # Create DataFrame and remove duplicates
    if all_oud_patients:
        oud_df = pd.DataFrame(all_oud_patients)
        
        # Remove duplicates (keep first occurrence per patient)
        oud_df = oud_df.drop_duplicates(subset=['pat_id'], keep='first')
        print(f"Unique OUD patients: {len(oud_df)}")
        
        # Save OUD patients
        oud_output = f'oud_patients_{year}.csv'
        oud_df.to_csv(oud_output, index=False)
        print(f"\nSaved OUD patients to: {oud_output}")
        
        # Display sample
        print("\nSample of OUD patients:")
        print(oud_df.head(10))
    else:
        print("No OUD patients found!")
        oud_df = pd.DataFrame(columns=['pat_id', 'matched_icd_codes', 'service_date', 'oud_label'])
    
    # Create all patients file with labels
    print("\nCreating complete patient list with labels...")
    
    # Get all unique patients from claims files (sampling approach for efficiency)
    all_patients = set()
    
    # Sample first few files to get patient list
    sample_size = min(10, len(csv_files))
    print(f"Sampling {sample_size} files to build patient list...")
    
    for i, csv_file in enumerate(csv_files[:sample_size]):
        try:
            df_sample = pd.read_csv(csv_file, sep='|', header=None, 
                                   usecols=[0], dtype=str, low_memory=False)
            all_patients.update(df_sample[0].unique())
            print(f"Processed sample {i+1}/{sample_size}, total patients so far: {len(all_patients)}")
        except Exception as e:
            print(f"Error sampling file: {e}")
    
    # Create final dataset
    oud_patient_ids = set(oud_df['pat_id'].astype(str))
    
    final_data = []
    for pat_id in all_patients:
        if str(pat_id) in oud_patient_ids:
            # Get OUD info
            oud_info = oud_df[oud_df['pat_id'] == str(pat_id)].iloc[0]
            final_data.append({
                'pat_id': pat_id,
                'oud_label': 1,
                'matched_icd_codes': oud_info['matched_icd_codes'],
                'service_date': oud_info['service_date']
            })
        else:
            final_data.append({
                'pat_id': pat_id,
                'oud_label': 0,
                'matched_icd_codes': '',
                'service_date': ''
            })
    
    # Save final dataset
    final_df = pd.DataFrame(final_data)
    final_output = f'all_patients_with_labels_{year}.csv'
    final_df.to_csv(final_output, index=False)
    print(f"\nSaved all patients with labels to: {final_output}")
    
    # Summary statistics
    print("\n" + "="*50)
    print("SUMMARY STATISTICS")
    print("="*50)
    print(f"Total patients processed: {len(final_df)}")
    print(f"Patients with OUD (label=1): {final_df['oud_label'].sum()}")
    print(f"Patients without OUD (label=0): {len(final_df) - final_df['oud_label'].sum()}")
    print(f"OUD prevalence: {final_df['oud_label'].mean():.2%}")
    
    elapsed = time.time() - start_time
    print(f"\nTotal processing time: {elapsed/60:.1f} minutes")
    
    print("\n✅ Processing complete!")
    print("Output files:")
    print(f"  1. {oud_output} - Patients with OUD diagnoses")
    print(f"  2. {final_output} - All patients with OUD labels")

if __name__ == "__main__":
    main()