"""
Extract OUD labels from IQVIA claims files by matching ICD codes.
Uses the same file access approach as 0_filter_data.py through read_iqvia module.

Input:
- extracted_icd_codes.csv: List of OUD-related ICD codes (with X/x as wildcards)
- IQVIA claims files accessed through read_iqvia module

Output:
- oud_patients_2006.csv: Patients with OUD (pat_id, matched_icd_codes, oud_label=1)
- all_patients_with_labels_2006.csv: All patients with their OUD labels (0 or 1)
"""

import pandas as pd
import numpy as np
import os
import re
import time
from multiprocessing import Pool, cpu_count
from tqdm import tqdm
import sys

# Add the current directory to Python path to import read_iqvia
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# Import the read_iqvia module that 0_filter_data.py uses
try:
    from read_iqvia import read_header
except ImportError:
    print("Warning: Could not import read_iqvia module, using fallback method")
    read_header = None

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

def get_header_for_year(year='2006'):
    """Get header using the same method as read_iqvia.py"""
    
    # First try using the imported function
    if read_header is not None:
        try:
            header_data = read_header()
            if year in header_data:
                return header_data[year]
        except Exception as e:
            print(f"Error using read_header: {e}")
    
    # Fallback: read directly
    header_folder = '/sharefolder/IQVIA/header'
    header_file = f'header_claims_{year}.csv'
    
    # Try local copy first
    local_paths = [
        f'./{header_file}',
        f'./header/{header_file}',
        os.path.join(header_folder, header_file)
    ]
    
    for path in local_paths:
        if os.path.exists(path):
            try:
                with open(path, 'r') as f:
                    header = f.readline().strip().split('|')
                print(f"Loaded header from: {path}")
                return header
            except Exception as e:
                continue
    
    # Default header based on data dictionary
    print("Using default header based on data dictionary...")
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

def process_csv_part(args):
    """Process a single CSV part file"""
    file_path, header, patterns, part_num, total_parts = args
    
    print(f"Processing part {part_num}/{total_parts}...", flush=True)
    
    try:
        # Read CSV file - same approach as read_iqvia_claims
        data_part = pd.read_csv(file_path, sep='|', header=None, dtype=str)
        data_part.columns = header
        
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
        
        print(f"Part {part_num}: Found {len(oud_patients)} OUD patients", flush=True)
        return oud_patients
        
    except Exception as e:
        print(f"Error processing part {part_num}: {e}", flush=True)
        return []

def extract_oud_labels_year(year='2006'):
    """Extract OUD labels for a specific year using the same approach as 0_filter_data.py"""
    
    print(f"\nProcessing year {year}...")
    
    # Load ICD patterns
    patterns, original_codes = load_icd_codes('extracted_icd_codes.csv')
    print(f"Sample ICD patterns: {patterns[:5]}")
    
    # Get header
    header = get_header_for_year(year)
    print(f"Header columns ({len(header)}): {header[:10]}...")
    
    # Get list of claims files using os.listdir (same as read_iqvia.py)
    claims_folder = '/sharefolder/IQVIA'  
    year_folder = os.path.join(claims_folder, f'claims_{year}')
    csv_in_parts_folder = os.path.join(year_folder, 'csv_in_parts')
    
    if not os.path.exists(csv_in_parts_folder):
        print(f"ERROR: Directory not found: {csv_in_parts_folder}")
        return None
    
    try:
        # Use os.listdir instead of glob
        csv_files = [file for file in os.listdir(csv_in_parts_folder) if file.endswith('.csv')]
        csv_files.sort()  # Sort for consistent ordering
        
        # Create full paths
        csv_file_paths = [os.path.join(csv_in_parts_folder, file) for file in csv_files]
        
        print(f"Found {len(csv_files)} CSV files to process")
        
        if not csv_files:
            print("No CSV files found!")
            return None
            
        # Show first few files
        print(f"First few files: {csv_files[:3]}")
        
    except PermissionError as e:
        print(f"Permission denied accessing directory: {e}")
        return None
    except Exception as e:
        print(f"Error listing directory: {e}")
        return None
    
    # Process files using multiprocessing
    args_list = [(f, header, patterns, i+1, len(csv_file_paths)) 
                 for i, f in enumerate(csv_file_paths)]
    
    all_oud_patients = []
    
    # Use fewer processes to avoid permission issues
    with Pool(processes=min(4, cpu_count())) as pool:
        results = list(tqdm(
            pool.imap(process_csv_part, args_list),
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
        oud_df.to_csv(output_path, index=False)
        print(f"Saved OUD patients to: {output_path}")
        
        return oud_df
    else:
        print("No OUD patients found")
        return pd.DataFrame()

def main():
    """Main function"""
    start_time = time.time()
    
    print("="*80)
    print("OUD LABEL EXTRACTION FROM CLAIMS DATA")
    print("Using read_iqvia module approach")
    print("="*80)
    
    year = '2006'
    
    try:
        # Extract OUD patients
        oud_df = extract_oud_labels_year(year)
        
        if oud_df is not None and len(oud_df) > 0:
            # Create final dataset with labels
            print("\nCreating labeled dataset...")
            
            # Load the feature dataset to get all patients
            features_path = '/sharefolder/wanglab/MME/final_features.csv'
            if os.path.exists(features_path):
                features_df = pd.read_csv(features_path)
                print(f"Loaded {len(features_df)} patients from features file")
                
                # Add OUD labels
                oud_patients = set(oud_df['pat_id'].astype(str))
                features_df['oud_label'] = features_df['pat_id'].astype(str).apply(
                    lambda x: 1 if x in oud_patients else 0
                )
                
                # Save final labeled dataset
                output_path = f'/sharefolder/wanglab/MME/final_dataset_with_labels_{year}.csv'
                features_df.to_csv(output_path, index=False)
                print(f"Saved final labeled dataset to: {output_path}")
                
                # Summary
                print(f"\nSummary:")
                print(f"Total patients: {len(features_df)}")
                print(f"OUD patients: {features_df['oud_label'].sum()}")
                print(f"OUD prevalence: {features_df['oud_label'].mean():.2%}")
        
    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()
    
    elapsed = time.time() - start_time
    print(f"\nTotal time: {elapsed/60:.1f} minutes")

if __name__ == "__main__":
    main()