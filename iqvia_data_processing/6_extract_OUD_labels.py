"""
Extract OUD labels from IQVIA claims files by matching ICD codes.
Fixed version with proper permission handling based on IST feedback.

Input:
- extracted_icd_codes.csv: List of OUD-related ICD codes
- IQVIA claims files accessed in read-only mode

Output:
- oud_patients_2006.csv: Patients with OUD diagnoses
"""

import pandas as pd
import numpy as np
import os
import re
import time
from multiprocessing import Pool, cpu_count
from tqdm import tqdm
import sys

def debug_permissions():
    """Debug function to check user context and permissions"""
    print("="*80)
    print("PERMISSION DEBUG INFORMATION")
    print("="*80)
    print(f"Current User ID (UID): {os.getuid()}")
    print(f"Current Group ID (GID): {os.getgid()}")
    print(f"Expected UID: 1019044820")
    
    if os.getuid() != 1019044820:
        print("WARNING: Running under unexpected user context!")
    else:
        print("✓ Running under correct user context")
    
    print(f"Current working directory: {os.getcwd()}")
    print(f"Python executable: {sys.executable}")
    print("="*80)

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
                # Open in read-only mode explicitly
                with open(path, 'r') as f:
                    icd_df = pd.read_csv(f)
                print(f"Loaded ICD codes from: {path}")
                break
            except Exception as e:
                print(f"Error reading {path}: {e}")
    
    if icd_df is None:
        raise FileNotFoundError(f"Could not find {filepath}")
    
    # Get unique ICD codes
    icd_codes = icd_df['ICD_Code'].unique().tolist()
    print(f"Found {len(icd_codes)} unique ICD codes")
    
    # Convert to regex patterns
    patterns = []
    for code in icd_codes:
        pattern = str(code).replace('X', r'\d').replace('x', r'\d')
        pattern = re.sub(r'([.+?^${}()|[\]])', r'\\\1', pattern)
        pattern = pattern.replace(r'\\d', r'\d')
        patterns.append(f'^{pattern}$')
    
    return patterns, icd_codes

def safe_read_header(year='2006'):
    """Safely read header file with explicit read-only access"""
    header_file = f'header_claims_{year}.csv'
    
    # Try local paths first, then IQVIA directory
    paths_to_try = [
        f'./{header_file}',
        f'./header/{header_file}',
        f'/sharefolder/IQVIA/header/{header_file}'
    ]
    
    for path in paths_to_try:
        if os.path.exists(path):
            try:
                # Check read permissions explicitly
                if os.access(path, os.R_OK):
                    print(f"✓ Have read access to: {path}")
                    # Open in read-only mode
                    with open(path, 'r') as f:
                        header = f.readline().strip().split('|')
                    print(f"Successfully loaded header with {len(header)} columns")
                    return header
                else:
                    print(f"✗ No read access to: {path}")
            except Exception as e:
                print(f"Error reading {path}: {e}")
    
    # Default header if file not accessible
    print("Using default header based on data dictionary...")
    return [
        'pat_id', 'claimno', 'linenum', 'rectype', 'tos_flag', 'pos', 
        'conf_num', 'patstat', 'billtype', 'ndc', 'from_dt', 'to_dt',
        'diag1', 'diag2', 'diag3', 'diag4', 'diag5', 'proc1', 'proc2',
        'dayssup', 'qty', 'copay', 'pay_typ', 'prscbr_id', 'prscbr_gender',
        'prscbr_yob', 'prscbr_zip3', 'specialty', 'pharm_blk', 'pharm_zip3'
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

def process_csv_part_readonly(args):
    """Process a single CSV part file with read-only access"""
    file_path, header, patterns, part_num, total_parts = args
    
    if part_num % 10 == 0:  # Print every 10th part
        print(f"Processing part {part_num}/{total_parts}...", flush=True)
    
    try:
        # Check read access before attempting to open
        if not os.access(file_path, os.R_OK):
            print(f"No read access to {file_path}")
            return []
        
        # Read CSV file in read-only mode
        with open(file_path, 'rb') as f:  # Binary mode for safer reading
            data_part = pd.read_csv(f, sep='|', header=None, dtype=str)
        
        # Assign columns
        if len(data_part.columns) == len(header):
            data_part.columns = header
        else:
            print(f"Warning: Column mismatch in part {part_num}")
            print(f"Expected {len(header)} columns, got {len(data_part.columns)}")
            return []
        
        # Diagnosis columns to check
        diag_cols = ['diag1', 'diag2', 'diag3', 'diag4', 'diag5']
        available_diag_cols = [col for col in diag_cols if col in data_part.columns]
        
        if not available_diag_cols:
            print(f"Warning: No diagnosis columns found in part {part_num}")
            return []
        
        # Find patients with OUD codes
        oud_patients = []
        
        # Process in chunks for memory efficiency
        chunk_size = 10000
        for start_idx in range(0, len(data_part), chunk_size):
            chunk = data_part.iloc[start_idx:start_idx + chunk_size]
            
            for _, row in chunk.iterrows():
                matched_codes = []
                
                # Check each diagnosis column
                for col in available_diag_cols:
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
        
        if oud_patients and part_num % 10 == 0:
            print(f"Part {part_num}: Found {len(oud_patients)} OUD patients", flush=True)
        
        return oud_patients
        
    except Exception as e:
        print(f"Error processing part {part_num}: {e}", flush=True)
        return []

def safe_list_csv_files(directory):
    """Safely list CSV files in directory with read-only access"""
    csv_files = []
    
    try:
        # Check if we can access the directory
        if not os.access(directory, os.R_OK | os.X_OK):
            print(f"No read/execute access to directory: {directory}")
            return []
        
        # List files
        for filename in os.listdir(directory):
            if filename.endswith('.csv'):
                filepath = os.path.join(directory, filename)
                # Check if we can read each file
                if os.access(filepath, os.R_OK):
                    csv_files.append(filepath)
                else:
                    print(f"Skipping {filename} - no read access")
        
        csv_files.sort()
        return csv_files
        
    except PermissionError as e:
        print(f"Permission denied: {e}")
        return []
    except Exception as e:
        print(f"Error listing directory: {e}")
        return []

def extract_oud_labels_year(year='2006'):
    """Extract OUD labels with proper permission handling"""
    
    print(f"\nProcessing year {year}...")
    
    # Debug permissions first
    debug_permissions()
    
    # Load ICD patterns
    patterns, original_codes = load_icd_codes('extracted_icd_codes.csv')
    print(f"Loaded {len(patterns)} ICD patterns")
    
    # Get header safely
    header = safe_read_header(year)
    
    # Define paths
    claims_folder = '/sharefolder/IQVIA'  
    year_folder = os.path.join(claims_folder, f'claims_{year}')
    csv_in_parts_folder = os.path.join(year_folder, 'csv_in_parts')
    
    print(f"\nChecking access to: {csv_in_parts_folder}")
    
    # Check directory access
    if not os.path.exists(csv_in_parts_folder):
        print(f"ERROR: Directory not found: {csv_in_parts_folder}")
        return None
    
    if not os.access(csv_in_parts_folder, os.R_OK | os.X_OK):
        print(f"ERROR: No read/execute access to: {csv_in_parts_folder}")
        return None
    
    # Get list of CSV files safely
    csv_file_paths = safe_list_csv_files(csv_in_parts_folder)
    
    if not csv_file_paths:
        print("No accessible CSV files found!")
        return None
    
    print(f"Found {len(csv_file_paths)} accessible CSV files")
    print(f"First file: {os.path.basename(csv_file_paths[0])}")
    
    # Process files using multiprocessing
    args_list = [(f, header, patterns, i+1, len(csv_file_paths)) 
                 for i, f in enumerate(csv_file_paths)]
    
    all_oud_patients = []
    
    # Use fewer processes to avoid permission issues
    num_processes = min(4, cpu_count())
    print(f"\nProcessing with {num_processes} parallel processes...")
    
    with Pool(processes=num_processes) as pool:
        results = list(tqdm(
            pool.imap(process_csv_part_readonly, args_list),
            total=len(csv_file_paths),
            desc=f"Processing {year} claims"
        ))
        
        # Combine results
        for result in results:
            if result:
                all_oud_patients.extend(result)
    
    print(f"\nTotal OUD records found: {len(all_oud_patients)}")
    
    if all_oud_patients:
        # Create DataFrame and remove duplicates
        oud_df = pd.DataFrame(all_oud_patients)
        oud_df = oud_df.drop_duplicates(subset=['pat_id'], keep='first')
        print(f"Unique OUD patients: {len(oud_df)}")
        
        # Save results
        output_path = f'/sharefolder/wanglab/MME/oud_patients_{year}.csv'
        try:
            oud_df.to_csv(output_path, index=False)
            print(f"Saved OUD patients to: {output_path}")
        except Exception as e:
            print(f"Error saving output: {e}")
            # Try saving to local directory
            local_output = f'./oud_patients_{year}.csv'
            oud_df.to_csv(local_output, index=False)
            print(f"Saved to local directory: {local_output}")
        
        return oud_df
    else:
        print("No OUD patients found")
        return pd.DataFrame()

def main():
    """Main function"""
    start_time = time.time()
    
    print("="*80)
    print("OUD LABEL EXTRACTION FROM CLAIMS DATA")
    print("Fixed version with proper permission handling")
    print("="*80)
    
    year = '2006'
    
    try:
        # Extract OUD patients
        oud_df = extract_oud_labels_year(year)
        
        if oud_df is not None and len(oud_df) > 0:
            # Create final dataset with labels
            print("\nCreating labeled dataset...")
            
            # Load the feature dataset
            features_path = '/sharefolder/wanglab/MME/final_features.csv'
            if os.path.exists(features_path) and os.access(features_path, os.R_OK):
                with open(features_path, 'r') as f:
                    features_df = pd.read_csv(f)
                print(f"Loaded {len(features_df)} patients from features file")
                
                # Add OUD labels
                oud_patients = set(oud_df['pat_id'].astype(str))
                features_df['oud_label'] = features_df['pat_id'].astype(str).apply(
                    lambda x: 1 if x in oud_patients else 0
                )
                
                # Save final labeled dataset
                output_path = f'/sharefolder/wanglab/MME/final_dataset_with_labels_{year}.csv'
                try:
                    features_df.to_csv(output_path, index=False)
                    print(f"Saved final labeled dataset to: {output_path}")
                except:
                    local_output = f'./final_dataset_with_labels_{year}.csv'
                    features_df.to_csv(local_output, index=False)
                    print(f"Saved to local directory: {local_output}")
                
                # Summary
                print(f"\nSummary:")
                print(f"Total patients: {len(features_df)}")
                print(f"OUD patients: {features_df['oud_label'].sum()}")
                print(f"OUD prevalence: {features_df['oud_label'].mean():.2%}")
            else:
                print(f"Cannot read features file: {features_path}")
        
    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()
    
    elapsed = time.time() - start_time
    print(f"\nTotal time: {elapsed/60:.1f} minutes")

if __name__ == "__main__":
    main()