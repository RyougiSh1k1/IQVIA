"""
Step 7.5: Extract Demographic Features
This script extracts age, gender, zip3 from enrollment synthetic data
and payment type from enrollment files, then merges with existing features.

Input files:
- /sharefolder/IQVIA/enroll_synth/csv_in_parts/*.csv (age, gender, zip3)
- /sharefolder/IQVIA/enroll2_{year}/csv_in_parts/*.csv (payment type)
- /sharefolder/wanglab/MME/final_dataset_with_oud_labels.csv (existing features)

Output files:
- /sharefolder/wanglab/MME/demographic_features.csv (extracted demographics)
- /sharefolder/wanglab/MME/final_dataset_with_all_features.csv (complete dataset)
"""

import pandas as pd
import numpy as np
import os
import time
from glob import glob
from multiprocessing import Pool, cpu_count
from tqdm import tqdm
from datetime import datetime

def load_header(header_type, year=None):
    """Load column headers for enrollment files"""
    if header_type == 'enroll_synth':
        header_files = [
            '/sharefolder/IQVIA/header/header_enroll_synth.csv',
            './header/header_enroll_synth.csv'
        ]
    elif header_type == 'enroll2' and year:
        header_files = [
            f'/sharefolder/IQVIA/header/header_enroll2_{year}.csv',
            f'./header/header_enroll2_{year}.csv'
        ]
    else:
        raise ValueError(f"Invalid header type: {header_type}")
    
    for path in header_files:
        if os.path.exists(path) and os.access(path, os.R_OK):
            try:
                with open(path, 'r') as f:
                    headers = f.readline().strip().split('|')
                print(f"Loaded header with {len(headers)} columns from {path}")
                return headers
            except Exception as e:
                print(f"Error reading {path}: {e}")
    
    # Default headers based on IQVIA documentation
    if header_type == 'enroll_synth':
        return ['pat_id', 'der_gender', 'der_yob', 'pat_zip3', 'enroll_start', 'enroll_end']
    else:
        return ['pat_id', 'pay_type', 'enroll_start', 'enroll_end', 'year']

def calculate_age(yob, reference_year=2022):
    """Calculate age from year of birth"""
    try:
        if pd.isna(yob) or yob == '' or yob == '0':
            return np.nan
        yob_int = int(float(yob))
        if yob_int < 1900 or yob_int > reference_year:
            return np.nan
        return reference_year - yob_int
    except:
        return np.nan

def process_enroll_synth_file(args):
    """Process a single enrollment synthetic file for demographics"""
    file_path, headers, file_num, total_files = args
    
    try:
        # Read CSV file
        df = pd.read_csv(file_path, sep='|', header=None, dtype=str, low_memory=False)
        
        # Assign headers
        if len(df.columns) == len(headers):
            df.columns = headers
        else:
            # Try to map common columns
            expected_cols = ['pat_id', 'der_gender', 'der_yob', 'pat_zip3']
            if len(df.columns) >= 4:
                df.columns = expected_cols + [f'col_{i}' for i in range(4, len(df.columns))]
            else:
                return []
        
        # Extract demographic features
        demographics = []
        
        for _, row in df.iterrows():
            pat_id = str(row.get('pat_id', '')).strip()
            if not pat_id:
                continue
                
            # Gender (M/F to 1/0)
            gender = str(row.get('der_gender', '')).strip().upper()
            gender_encoded = 1 if gender == 'M' else (0 if gender == 'F' else np.nan)
            
            # Age from year of birth
            yob = row.get('der_yob', '')
            age = calculate_age(yob)
            
            # 3-digit zip code
            zip3 = str(row.get('pat_zip3', '')).strip()
            if len(zip3) != 3 or not zip3.isdigit():
                zip3 = np.nan
            
            demographics.append({
                'pat_id': pat_id,
                'gender': gender_encoded,
                'age': age,
                'zip3': zip3
            })
        
        if file_num % 10 == 0:
            print(f"Processed file {file_num}/{total_files}: {len(demographics)} patients")
        
        return demographics
        
    except Exception as e:
        print(f"Error processing {file_path}: {e}")
        return []

def process_enroll2_file(args):
    """Process enrollment files for payment type"""
    file_path, headers, year, file_num, total_files = args
    
    try:
        # Read CSV file
        df = pd.read_csv(file_path, sep='|', header=None, dtype=str, low_memory=False)
        
        # Assign headers
        if len(df.columns) == len(headers):
            df.columns = headers
        else:
            # Map essential columns
            if len(df.columns) >= 2:
                df.columns = ['pat_id', 'pay_type'] + [f'col_{i}' for i in range(2, len(df.columns))]
            else:
                return []
        
        # Extract payment type
        payment_data = []
        
        for _, row in df.iterrows():
            pat_id = str(row.get('pat_id', '')).strip()
            pay_type = str(row.get('pay_type', '')).strip()
            
            if pat_id and pay_type:
                payment_data.append({
                    'pat_id': pat_id,
                    'pay_type': pay_type,
                    'year': year
                })
        
        return payment_data
        
    except Exception as e:
        print(f"Error processing {file_path}: {e}")
        return []

def extract_demographic_features():
    """Extract all demographic features"""
    start_time = time.time()
    
    print("="*80)
    print("EXTRACTING DEMOGRAPHIC FEATURES")
    print("="*80)
    
    # Step 1: Extract from enrollment synthetic data
    print("\n1. Processing enrollment synthetic data (age, gender, zip3)...")
    
    enroll_synth_dir = '/sharefolder/IQVIA/enroll_synth/csv_in_parts'
    if not os.path.exists(enroll_synth_dir):
        print(f"Error: Directory not found: {enroll_synth_dir}")
        return None
    
    # Get list of CSV files
    synth_files = sorted(glob(os.path.join(enroll_synth_dir, '*.csv')))
    accessible_files = [f for f in synth_files if os.access(f, os.R_OK)]
    print(f"Found {len(accessible_files)} accessible files")
    
    # Load header
    synth_headers = load_header('enroll_synth')
    
    # Process files in parallel
    args_list = [(f, synth_headers, i+1, len(accessible_files)) 
                 for i, f in enumerate(accessible_files)]
    
    all_demographics = []
    num_processes = min(8, cpu_count())
    
    with Pool(processes=num_processes) as pool:
        results = list(tqdm(
            pool.imap(process_enroll_synth_file, args_list),
            total=len(accessible_files),
            desc="Processing synthetic enrollment files"
        ))
        
        for result in results:
            if result:
                all_demographics.extend(result)
    
    print(f"Extracted demographics for {len(all_demographics)} records")
    
    # Convert to DataFrame and aggregate by patient
    demo_df = pd.DataFrame(all_demographics)
    demo_df = demo_df.groupby('pat_id').agg({
        'gender': 'first',
        'age': 'first',
        'zip3': 'first'
    }).reset_index()
    
    print(f"Unique patients with demographics: {len(demo_df)}")
    
    # Step 2: Extract payment type from enrollment files
    print("\n2. Processing enrollment files for payment type...")
    
    payment_data_all = []
    years = [str(y) for y in range(2006, 2023)]
    
    for year in tqdm(years, desc="Processing years"):
        enroll2_dir = f'/sharefolder/IQVIA/enroll2_{year}/csv_in_parts'
        
        if not os.path.exists(enroll2_dir):
            print(f"Skipping year {year} - directory not found")
            continue
        
        # Get CSV files for this year
        year_files = sorted(glob(os.path.join(enroll2_dir, '*.csv')))
        accessible_year_files = [f for f in year_files if os.access(f, os.R_OK)]
        
        if not accessible_year_files:
            continue
        
        # Load header for this year
        enroll2_headers = load_header('enroll2', year)
        
        # Process files for this year
        args_list = [(f, enroll2_headers, year, i+1, len(accessible_year_files)) 
                     for i, f in enumerate(accessible_year_files)]
        
        with Pool(processes=num_processes) as pool:
            results = pool.map(process_enroll2_file, args_list)
            
            for result in results:
                if result:
                    payment_data_all.extend(result)
    
    print(f"Extracted payment type for {len(payment_data_all)} records")
    
    # Convert to DataFrame and get most common payment type per patient
    if payment_data_all:
        payment_df = pd.DataFrame(payment_data_all)
        
        # Get most frequent payment type per patient
        payment_summary = payment_df.groupby(['pat_id', 'pay_type']).size().reset_index(name='count')
        payment_summary = payment_summary.sort_values(['pat_id', 'count'], ascending=[True, False])
        payment_summary = payment_summary.groupby('pat_id').first().reset_index()
        payment_summary = payment_summary[['pat_id', 'pay_type']]
        
        print(f"Unique patients with payment type: {len(payment_summary)}")
    else:
        payment_summary = pd.DataFrame(columns=['pat_id', 'pay_type'])
    
    # Step 3: Merge all demographic features
    print("\n3. Merging demographic features...")
    
    # Merge demographics with payment type
    if not payment_summary.empty:
        final_demo_df = demo_df.merge(payment_summary, on='pat_id', how='outer')
    else:
        final_demo_df = demo_df.copy()
        final_demo_df['pay_type'] = np.nan
    
    # Encode payment type
    payment_type_mapping = {
        '1': 'commercial',
        '2': 'medicare',
        '3': 'medicaid',
        '4': 'cash',
        '5': 'other'
    }
    
    final_demo_df['payment_type'] = final_demo_df['pay_type'].map(payment_type_mapping)
    final_demo_df = final_demo_df.drop('pay_type', axis=1)
    
    # Save demographic features
    output_path = '/sharefolder/wanglab/MME/demographic_features.csv'
    final_demo_df.to_csv(output_path, index=False)
    print(f"\nSaved demographic features to: {output_path}")
    
    # Print summary statistics
    print("\nDemographic Feature Summary:")
    print(f"Total unique patients: {len(final_demo_df)}")
    print(f"Patients with gender: {final_demo_df['gender'].notna().sum()}")
    print(f"Patients with age: {final_demo_df['age'].notna().sum()}")
    print(f"Patients with zip3: {final_demo_df['zip3'].notna().sum()}")
    print(f"Patients with payment type: {final_demo_df['payment_type'].notna().sum()}")
    
    if final_demo_df['age'].notna().any():
        print(f"\nAge statistics:")
        print(f"  Mean: {final_demo_df['age'].mean():.1f}")
        print(f"  Median: {final_demo_df['age'].median():.1f}")
        print(f"  Range: {final_demo_df['age'].min():.0f} - {final_demo_df['age'].max():.0f}")
    
    if final_demo_df['gender'].notna().any():
        gender_dist = final_demo_df['gender'].value_counts()
        print(f"\nGender distribution:")
        print(f"  Male (1): {gender_dist.get(1, 0)}")
        print(f"  Female (0): {gender_dist.get(0, 0)}")
    
    if final_demo_df['payment_type'].notna().any():
        print(f"\nPayment type distribution:")
        for ptype, count in final_demo_df['payment_type'].value_counts().items():
            print(f"  {ptype}: {count}")
    
    elapsed = time.time() - start_time
    print(f"\nExtraction completed in {elapsed/60:.1f} minutes")
    
    return final_demo_df

def merge_with_existing_features():
    """Merge demographic features with existing dataset"""
    print("\n" + "="*80)
    print("MERGING DEMOGRAPHIC FEATURES WITH EXISTING DATASET")
    print("="*80)
    
    # Load existing dataset
    existing_path = '/sharefolder/wanglab/MME/final_dataset_with_oud_labels.csv'
    demo_path = '/sharefolder/wanglab/MME/demographic_features.csv'
    
    if not os.path.exists(existing_path):
        print(f"Error: Existing dataset not found: {existing_path}")
        return False
    
    if not os.path.exists(demo_path):
        print(f"Error: Demographic features not found: {demo_path}")
        return False
    
    try:
        print("Loading existing dataset...")
        existing_df = pd.read_csv(existing_path)
        existing_df['pat_id'] = existing_df['pat_id'].astype(str)
        print(f"Existing dataset: {len(existing_df)} rows")
        
        print("Loading demographic features...")
        demo_df = pd.read_csv(demo_path)
        demo_df['pat_id'] = demo_df['pat_id'].astype(str)
        print(f"Demographic features: {len(demo_df)} patients")
        
        # Merge datasets
        print("Merging datasets...")
        final_df = existing_df.merge(demo_df, on='pat_id', how='left')
        
        # Create dummy variables for payment type
        if 'payment_type' in final_df.columns:
            payment_dummies = pd.get_dummies(final_df['payment_type'], prefix='payment', dummy_na=True)
            final_df = pd.concat([final_df, payment_dummies], axis=1)
        
        # Save final dataset
        output_path = '/sharefolder/wanglab/MME/final_dataset_with_all_features.csv'
        final_df.to_csv(output_path, index=False)
        print(f"\nSaved final dataset with all features to: {output_path}")
        
        # Print feature summary
        print("\nFinal Dataset Summary:")
        print(f"Total rows: {len(final_df)}")
        print(f"Total columns: {len(final_df.columns)}")
        
        # Check completeness of new features
        print("\nFeature completeness:")
        for col in ['gender', 'age', 'zip3']:
            if col in final_df.columns:
                completeness = (final_df[col].notna().sum() / len(final_df)) * 100
                print(f"  {col}: {completeness:.1f}% complete")
        
        # Payment type completeness
        payment_cols = [col for col in final_df.columns if col.startswith('payment_')]
        if payment_cols:
            has_payment = final_df[payment_cols].sum(axis=1) > 0
            payment_completeness = (has_payment.sum() / len(final_df)) * 100
            print(f"  payment_type: {payment_completeness:.1f}% complete")
        
        # List all features
        print("\nAll features in final dataset:")
        feature_cols = [col for col in final_df.columns 
                       if col not in ['pat_id', 'most_recent_date', 'first_oud_date', 'oud_year']]
        
        print(f"\nOriginal MME/Prescriber features:")
        mme_features = ['MME_last_365_days', 'MME_last_2_years', 'MME_prior_1_year', 
                       'MME_120_2_years', 'prscbr_last_2_years', 'prscrbr_last_180_days']
        for f in mme_features:
            if f in feature_cols:
                print(f"  - {f}")
        
        print(f"\nDemographic features:")
        demo_features = ['gender', 'age', 'zip3'] + payment_cols
        for f in demo_features:
            if f in feature_cols:
                print(f"  - {f}")
        
        print(f"\nTarget variable:")
        print(f"  - oud_label")
        
        return True
        
    except Exception as e:
        print(f"Error during merge: {e}")
        import traceback
        traceback.print_exc()
        return False

def main():
    """Main function"""
    print(" Starting Step 7.5: Extract and Merge Demographic Features")
    print("This process may take several hours due to the large data volume")
    
    # Extract demographic features
    demo_df = extract_demographic_features()
    
    if demo_df is not None and not demo_df.empty:
        # Merge with existing features
        if merge_with_existing_features():
            print("\n All demographic features successfully added!")
            print("\nNext steps:")
            print("1. Use 'final_dataset_with_all_features.csv' for ML/NN training")
            print("2. Update your ML scripts to include the new features")
            print("3. Consider handling missing values in demographic features")
        else:
            print("\n Failed to merge demographic features")
    else:
        print("\n Failed to extract demographic features")

if __name__ == "__main__":
    main()