"""
Step 6: Extract OUD Labels from All Years (2006-2022)
This script processes claims data from all years to identify patients with Opioid Use Disorder.
Uses multiprocessing for both year-level and file-level parallelization.

Input:
- extracted_icd_codes.csv: List of OUD-related ICD codes
- /sharefolder/IQVIA/claims_{year}/csv_in_parts/*.csv: Claims data files for each year

Output:
- oud_patients_all_years.csv: All patients with OUD diagnoses across all years
- oud_patients_{year}.csv: Year-specific OUD patient files (intermediate)
"""

import pandas as pd
import numpy as np
import os
import re
import time
from glob import glob
from multiprocessing import Pool, cpu_count
from tqdm import tqdm
from functools import partial

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
    
    return patterns

def load_header(year):
    """Load column headers for claims files"""
    header_path = f'/home/qinyu@chapman.edu/IQVIA/iqvia_data_processing/header/header_claims_{year}.csv'
    
    # Alternative paths
    alt_paths = [
        f'./header/header_claims_{year}.csv',
        f'/sharefolder/IQVIA/header/header_claims_{year}.csv'
    ]
    
    for path in [header_path] + alt_paths:
        if os.path.exists(path) and os.access(path, os.R_OK):
            try:
                with open(path, 'r') as f:
                    headers = f.readline().strip().split(',')
                    headers = [h.strip('"') for h in headers]
                print(f"Successfully loaded header with {len(headers)} columns")
                return headers
            except Exception as e:
                print(f"Error reading {path}: {e}")
    
    raise FileNotFoundError(f"Could not find header file for year {year}")

def process_csv_file(args):
    """Process a single CSV file to find OUD patients"""
    csv_path, header, patterns, file_num, total_files = args
    
    try:
        # Read the CSV file
        df = pd.read_csv(csv_path, sep='|', header=None, names=header, 
                        dtype=str, low_memory=False)
        
        # Find columns with ICD codes
        icd_columns = [col for col in df.columns if 'diag' in col.lower() or 'icd' in col.lower()]
        
        if not icd_columns:
            return []
        
        # Search for OUD ICD codes
        oud_records = []
        
        for _, row in df.iterrows():
            matched_codes = []
            
            # Check each ICD column
            for col in icd_columns:
                if pd.notna(row[col]):
                    icd_value = str(row[col]).strip()
                    # Check against all patterns
                    for pattern in patterns:
                        if re.match(pattern, icd_value):
                            matched_codes.append(icd_value)
                            break
            
            # If we found matching codes, record this patient
            if matched_codes:
                oud_records.append({
                    'pat_id': row['pat_id'],
                    'matched_icd_codes': ','.join(matched_codes),
                    'service_date': row.get('svcdate', ''),
                    'year': row.get('svcdate', '')[:4] if pd.notna(row.get('svcdate', '')) else ''
                })
        
        return oud_records
        
    except Exception as e:
        print(f"Error processing {csv_path}: {e}")
        return []

def process_year(year, patterns):
    """Process all claims files for a specific year"""
    print(f"\n{'='*60}")
    print(f"Processing Year {year}")
    print(f"{'='*60}")
    
    try:
        # Load header for this year
        headers = load_header(year)
        
        # Get list of CSV files
        claims_dir = f'/sharefolder/IQVIA/claims_{year}/csv_in_parts'
        
        if not os.path.exists(claims_dir):
            print(f"Claims directory not found: {claims_dir}")
            return pd.DataFrame()
        
        csv_files = sorted(glob(os.path.join(claims_dir, '*.csv')))
        
        # Only process files we have access to
        accessible_files = []
        for f in csv_files:
            if os.access(f, os.R_OK):
                accessible_files.append(f)
        
        print(f"Found {len(accessible_files)} accessible CSV files for year {year}")
        
        if not accessible_files:
            return pd.DataFrame()
        
        # Prepare arguments for multiprocessing
        args_list = [(f, headers, patterns, i+1, len(accessible_files)) 
                     for i, f in enumerate(accessible_files)]
        
        # Process files using multiprocessing
        all_oud_records = []
        num_processes = min(4, cpu_count())  # Limit processes per year
        
        with Pool(processes=num_processes) as pool:
            results = list(tqdm(
                pool.imap(process_csv_file, args_list),
                total=len(accessible_files),
                desc=f"Processing {year} files"
            ))
            
            # Combine results
            for result in results:
                if result:
                    all_oud_records.extend(result)
        
        print(f"Found {len(all_oud_records)} OUD records in year {year}")
        
        if all_oud_records:
            # Create DataFrame and remove duplicates
            year_df = pd.DataFrame(all_oud_records)
            year_df['year'] = year
            
            # Remove duplicates within this year
            year_df = year_df.drop_duplicates(subset=['pat_id'], keep='first')
            
            # Save year-specific results
            output_path = f'/sharefolder/wanglab/MME/oud_patients_{year}.csv'
            year_df.to_csv(output_path, index=False)
            print(f"Saved {len(year_df)} unique OUD patients for year {year}")
            
            return year_df
        else:
            return pd.DataFrame()
            
    except Exception as e:
        print(f"Error processing year {year}: {e}")
        return pd.DataFrame()

def combine_all_years(years):
    """Combine OUD patients from all years"""
    print("\n" + "="*80)
    print("COMBINING OUD PATIENTS FROM ALL YEARS")
    print("="*80)
    
    all_oud_dfs = []
    
    for year in years:
        year_file = f'/sharefolder/wanglab/MME/oud_patients_{year}.csv'
        if os.path.exists(year_file):
            try:
                df = pd.read_csv(year_file)
                all_oud_dfs.append(df)
                print(f"Loaded {len(df)} patients from year {year}")
            except Exception as e:
                print(f"Error loading {year_file}: {e}")
    
    if all_oud_dfs:
        # Combine all DataFrames
        combined_df = pd.concat(all_oud_dfs, ignore_index=True)
        print(f"\nTotal OUD records before deduplication: {len(combined_df)}")
        
        # Sort by service date to keep the earliest diagnosis
        combined_df['service_date'] = pd.to_datetime(combined_df['service_date'], errors='coerce')
        combined_df = combined_df.sort_values('service_date')
        
        # Remove duplicates, keeping first occurrence
        combined_df = combined_df.drop_duplicates(subset=['pat_id'], keep='first')
        print(f"Unique OUD patients across all years: {len(combined_df)}")
        
        # Add OUD label
        combined_df['oud_label'] = 1
        
        return combined_df
    else:
        return pd.DataFrame()

def main():
    """Main function to process all years"""
    start_time = time.time()
    
    print("="*80)
    print("OUD LABEL EXTRACTION FROM ALL YEARS (2006-2022)")
    print("Multiprocessing enabled for efficiency")
    print("="*80)
    
    # Load ICD patterns once
    try:
        patterns = load_icd_codes()
        print(f"Loaded {len(patterns)} ICD patterns")
    except Exception as e:
        print(f"Error loading ICD codes: {e}")
        return
    
    # Define years to process
    years = [str(y) for y in range(2006, 2023)]
    print(f"\nYears to process: {', '.join(years)}")
    
    # Option 1: Process years sequentially (more stable)
    year_results = []
    for year in years:
        result = process_year(year, patterns)
        if not result.empty:
            year_results.append(result)
    
    # Option 2: Process years in parallel (faster but may hit resource limits)
    # Uncomment below to use parallel year processing
    """
    num_year_processes = min(4, cpu_count() // 2)  # Conservative parallelization
    print(f"\nProcessing years with {num_year_processes} parallel processes...")
    
    with Pool(processes=num_year_processes) as pool:
        process_year_partial = partial(process_year, patterns=patterns)
        year_results = pool.map(process_year_partial, years)
    
    # Filter out empty results
    year_results = [r for r in year_results if not r.empty]
    """
    
    # Combine all years
    if year_results:
        combined_df = combine_all_years(years)
        
        if not combined_df.empty:
            # Save final combined results
            output_path = '/sharefolder/wanglab/MME/oud_patients_all_years.csv'
            combined_df.to_csv(output_path, index=False)
            print(f"\nSaved combined OUD patients to: {output_path}")
            
            # Summary statistics
            print("\n" + "="*50)
            print("FINAL SUMMARY STATISTICS")
            print("="*50)
            print(f"Total unique OUD patients: {len(combined_df)}")
            
            # Year distribution
            if 'year' in combined_df.columns:
                year_counts = combined_df['year'].value_counts().sort_index()
                print("\nOUD patients by year:")
                for year, count in year_counts.items():
                    print(f"  {year}: {count:,}")
            
            # Save summary
            summary_path = '/sharefolder/wanglab/MME/oud_extraction_summary.txt'
            with open(summary_path, 'w') as f:
                f.write(f"OUD Extraction Summary\n")
                f.write(f"=====================\n")
                f.write(f"Processing completed: {time.strftime('%Y-%m-%d %H:%M:%S')}\n")
                f.write(f"Total unique OUD patients: {len(combined_df)}\n")
                f.write(f"Years processed: {', '.join(years)}\n")
                f.write(f"Total processing time: {(time.time() - start_time)/60:.1f} minutes\n")
    
    elapsed = time.time() - start_time
    print(f"\nTotal processing time: {elapsed/3600:.1f} hours")
    print("\n OUD extraction complete!")

if __name__ == "__main__":
    main()