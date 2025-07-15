"""
Fix Demographic Features Extraction
This script correctly extracts age (from der_yob), gender (der_sex), and payment type (pay_type)
from IQVIA enrollment data and merges with existing features.

Input files:
- /sharefolder/IQVIA/enroll_synth/csv_in_parts/*.csv (der_yob, der_sex)
- /sharefolder/IQVIA/enroll2_{year}/csv_in_parts/*.csv (pay_type)
- /sharefolder/wanglab/MME/final_dataset_with_all_features.csv (existing dataset with zip3)

Output files:
- /sharefolder/wanglab/MME/demographic_features_fixed.csv (corrected demographics)
- /sharefolder/wanglab/MME/final_dataset_complete.csv (final dataset with all features)
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
            './header/header_enroll_synth.csv',
            '/home/qinyu@chapman.edu/IQVIA/iqvia_data_processing/header/header_enroll_synth.csv'
        ]
    elif header_type == 'enroll2' and year:
        header_files = [
            f'/sharefolder/IQVIA/header/header_enroll2_{year}.csv',
            f'./header/header_enroll2_{year}.csv',
            f'/home/qinyu@chapman.edu/IQVIA/iqvia_data_processing/header/header_enroll2_{year}.csv'
        ]
    else:
        raise ValueError(f"Invalid header type: {header_type}")
    
    for path in header_files:
        if os.path.exists(path) and os.access(path, os.R_OK):
            try:
                with open(path, 'r') as f:
                    headers = f.readline().strip().split('|')
                print(f"Loaded header with {len(headers)} columns from {path}")
                
                # Print relevant columns for debugging
                if header_type == 'enroll_synth':
                    for i, h in enumerate(headers):
                        if any(term in h.lower() for term in ['yob', 'sex', 'gender', 'der']):
                            print(f"  Column {i}: {h}")
                elif header_type == 'enroll2':
                    for i, h in enumerate(headers):
                        if any(term in h.lower() for term in ['pay', 'type']):
                            print(f"  Column {i}: {h}")
                
                return headers
            except Exception as e:
                print(f"Error reading {path}: {e}")
    
    print(f"Warning: Could not load header for {header_type}, using default mapping")
    return None

def calculate_age_from_yob(yob, reference_year=2022):
    """Calculate age from year of birth (der_yob)"""
    try:
        if pd.isna(yob) or yob == '' or yob == '0':
            return np.nan
        
        # Handle various formats of year of birth
        yob_str = str(yob).strip()
        
        # Extract 4-digit year
        if len(yob_str) >= 4:
            # Try to extract year from various formats
            if yob_str[:4].isdigit():
                yob_int = int(yob_str[:4])
            elif yob_str[-4:].isdigit():
                yob_int = int(yob_str[-4:])
            else:
                return np.nan
        else:
            return np.nan
        
        # Validate year
        if yob_int < 1900 or yob_int > reference_year:
            return np.nan
            
        age = reference_year - yob_int
        
        # Validate age
        if age < 0 or age > 120:
            return np.nan
            
        return age
        
    except Exception:
        return np.nan

def encode_gender(sex_value):
    """Encode gender from der_sex field"""
    try:
        if pd.isna(sex_value):
            return np.nan
            
        sex_str = str(sex_value).strip().upper()
        
        # Handle various encodings
        if sex_str in ['M', 'MALE', '1', 'M ']:
            return 1
        elif sex_str in ['F', 'FEMALE', '2', 'F ', '0']:
            return 0
        else:
            return np.nan
            
    except Exception:
        return np.nan

def process_enroll_synth_file_fixed(args):
    """Process enrollment synthetic file for der_yob and der_sex"""
    file_path, headers, file_num, total_files = args
    
    try:
        # Read CSV file
        df = pd.read_csv(file_path, sep='|', header=None, dtype=str, low_memory=False)
        
        # Try to find the correct columns
        if headers and len(df.columns) == len(headers):
            df.columns = headers
        
        # Find der_yob and der_sex columns
        yob_col = None
        sex_col = None
        
        for col in df.columns:
            col_lower = str(col).lower()
            if 'der_yob' in col_lower or 'deryob' in col_lower:
                yob_col = col
            elif 'der_sex' in col_lower or 'dersex' in col_lower:
                sex_col = col
        
        # If not found by exact match, try fuzzy matching
        if not yob_col:
            for col in df.columns:
                if 'yob' in str(col).lower():
                    yob_col = col
                    break
        
        if not sex_col:
            for col in df.columns:
                col_lower = str(col).lower()
                if 'sex' in col_lower or 'gender' in col_lower:
                    sex_col = col
                    break
        
        if file_num <= 5:  # Debug first few files
            print(f"\nFile {file_num}: {os.path.basename(file_path)}")
            print(f"  Columns found: {df.columns.tolist()[:10]}...")
            print(f"  YOB column: {yob_col}")
            print(f"  Sex column: {sex_col}")
            if yob_col and sex_col:
                print(f"  Sample YOB values: {df[yob_col].dropna().head(3).tolist()}")
                print(f"  Sample Sex values: {df[sex_col].dropna().head(3).tolist()}")
        
        if not yob_col and not sex_col:
            print(f"Warning: Could not find der_yob or der_sex columns in file {file_num}")
            return []
        
        # Extract demographics
        demographics = []
        
        for _, row in df.iterrows():
            pat_id = str(row.get('pat_id', '')).strip()
            if not pat_id or pat_id == 'nan':
                continue
            
            # Extract age from der_yob
            age = np.nan
            if yob_col:
                yob_value = row.get(yob_col, '')
                age = calculate_age_from_yob(yob_value)
            
            # Extract gender from der_sex
            gender = np.nan
            if sex_col:
                sex_value = row.get(sex_col, '')
                gender = encode_gender(sex_value)
            
            # Only add if we have at least one valid value
            if not (pd.isna(age) and pd.isna(gender)):
                demographics.append({
                    'pat_id': pat_id,
                    'age': age,
                    'gender': gender
                })
        
        if file_num % 10 == 0:
            print(f"Processed file {file_num}/{total_files}: {len(demographics)} valid records")
        
        return demographics
        
    except Exception as e:
        print(f"Error processing {file_path}: {e}")
        import traceback
        traceback.print_exc()
        return []

def process_enroll2_file_fixed(args):
    """Process enrollment file for pay_type"""
    file_path, headers, year, file_num, total_files = args
    
    try:
        # Read CSV file
        df = pd.read_csv(file_path, sep='|', header=None, dtype=str, low_memory=False)
        
        # Try to assign headers
        if headers and len(df.columns) == len(headers):
            df.columns = headers
        
        # Find pay_type column
        pay_type_col = None
        for col in df.columns:
            col_lower = str(col).lower()
            if 'pay_type' in col_lower or 'paytype' in col_lower or col_lower == 'pay_typ':
                pay_type_col = col
                break
        
        # If not found, try generic search
        if not pay_type_col:
            for col in df.columns:
                if 'pay' in str(col).lower() and 'typ' in str(col).lower():
                    pay_type_col = col
                    break
        
        if file_num <= 3:  # Debug first few files
            print(f"\nYear {year}, File {file_num}: {os.path.basename(file_path)}")
            print(f"  Pay type column: {pay_type_col}")
            if pay_type_col:
                print(f"  Sample values: {df[pay_type_col].dropna().head(5).tolist()}")
        
        if not pay_type_col:
            return []
        
        # Extract payment data
        payment_data = []
        
        for _, row in df.iterrows():
            pat_id = str(row.get('pat_id', '')).strip()
            pay_type = str(row.get(pay_type_col, '')).strip()
            
            if pat_id and pat_id != 'nan' and pay_type and pay_type != 'nan':
                payment_data.append({
                    'pat_id': pat_id,
                    'pay_type': pay_type,
                    'year': year
                })
        
        return payment_data
        
    except Exception as e:
        print(f"Error processing {file_path}: {e}")
        return []

def extract_demographic_features_fixed():
    """Extract age, gender, and payment type with correct column names"""
    start_time = time.time()
    
    print("="*80)
    print("EXTRACTING DEMOGRAPHIC FEATURES (FIXED)")
    print("Target columns: der_yob, der_sex, pay_type")
    print("="*80)
    
    # Step 1: Extract age and gender from enrollment synthetic data
    print("\n1. Processing enrollment synthetic data (der_yob, der_sex)...")
    
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
            pool.imap(process_enroll_synth_file_fixed, args_list),
            total=len(accessible_files),
            desc="Processing enrollment synthetic files"
        ))
        
        for result in results:
            if result:
                all_demographics.extend(result)
    
    print(f"\nExtracted demographics for {len(all_demographics)} records")
    
    # Convert to DataFrame and aggregate by patient
    if all_demographics:
        demo_df = pd.DataFrame(all_demographics)
        
        # Aggregate by patient, taking first non-null value
        demo_df = demo_df.groupby('pat_id').agg({
            'age': lambda x: x.dropna().iloc[0] if len(x.dropna()) > 0 else np.nan,
            'gender': lambda x: x.dropna().iloc[0] if len(x.dropna()) > 0 else np.nan
        }).reset_index()
        
        print(f"Unique patients with age/gender: {len(demo_df)}")
        print(f"  Patients with valid age: {demo_df['age'].notna().sum()}")
        print(f"  Patients with valid gender: {demo_df['gender'].notna().sum()}")
        
        # Print age distribution
        if demo_df['age'].notna().any():
            print(f"\nAge statistics:")
            print(f"  Mean: {demo_df['age'].mean():.1f}")
            print(f"  Median: {demo_df['age'].median():.1f}")
            print(f"  Min: {demo_df['age'].min():.0f}")
            print(f"  Max: {demo_df['age'].max():.0f}")
    else:
        demo_df = pd.DataFrame(columns=['pat_id', 'age', 'gender'])
    
    # Step 2: Extract payment type from enrollment files
    print("\n2. Processing enrollment files for pay_type...")
    
    payment_data_all = []
    years = [str(y) for y in range(2006, 2023)]
    
    for year in tqdm(years, desc="Processing years"):
        enroll2_dir = f'/sharefolder/IQVIA/enroll2_{year}/csv_in_parts'
        
        if not os.path.exists(enroll2_dir):
            print(f"  Skipping year {year} - directory not found")
            continue
        
        # Get CSV files for this year
        year_files = sorted(glob(os.path.join(enroll2_dir, '*.csv')))
        accessible_year_files = [f for f in year_files if os.access(f, os.R_OK)]
        
        if not accessible_year_files:
            continue
        
        print(f"\n  Year {year}: {len(accessible_year_files)} files")
        
        # Load header for this year
        enroll2_headers = load_header('enroll2', year)
        
        # Process files for this year
        args_list = [(f, enroll2_headers, year, i+1, len(accessible_year_files)) 
                     for i, f in enumerate(accessible_year_files)]
        
        with Pool(processes=num_processes) as pool:
            results = pool.map(process_enroll2_file_fixed, args_list)
            
            for result in results:
                if result:
                    payment_data_all.extend(result)
    
    print(f"\nExtracted payment type for {len(payment_data_all)} records")
    
    # Process payment type data
    payment_summary = pd.DataFrame(columns=['pat_id', 'payment_type'])
    
    if payment_data_all:
        payment_df = pd.DataFrame(payment_data_all)
        
        # Map payment codes to categories based on IQVIA data dictionary
        # pay_type values:
        # A=Medicare Part C
        # C=Commercial
        # K=State Children's Health Insurance Program (SCHIP)
        # M=Medicaid
        # R=Medicare Risk (presently known as Medicare Advantage)
        # S=Self-Insured
        # T=Medicare Cost (Medicare Supplemental)
        # U=Unknown/Missing
        # X=RX Only
        
        payment_mapping = {
            'A': 'medicare',      # Medicare Part C
            'C': 'commercial',    # Commercial
            'K': 'medicaid',      # SCHIP (children's insurance, similar to Medicaid)
            'M': 'medicaid',      # Medicaid
            'R': 'medicare',      # Medicare Risk/Advantage
            'S': 'self_insured',  # Self-Insured
            'T': 'medicare',      # Medicare Cost/Supplemental
            'U': 'unknown',       # Unknown/Missing
            'X': 'rx_only',       # RX Only
            # Handle lowercase variants
            'a': 'medicare',
            'c': 'commercial',
            'k': 'medicaid',
            'm': 'medicaid',
            'r': 'medicare',
            's': 'self_insured',
            't': 'medicare',
            'u': 'unknown',
            'x': 'rx_only'
        }
        
        # Clean pay_type values before mapping
        payment_df['pay_type_clean'] = payment_df['pay_type'].astype(str).str.strip().str.upper()
        
        # Apply mapping
        payment_df['payment_type'] = payment_df['pay_type_clean'].map(payment_mapping)
        
        # Handle unmapped values
        unmapped = payment_df[payment_df['payment_type'].isna() & payment_df['pay_type'].notna()]
        if len(unmapped) > 0:
            print(f"\nUnmapped payment types found:")
            unmapped_counts = unmapped['pay_type_clean'].value_counts().head(20)
            print(unmapped_counts)
            
            # Map any remaining unmapped non-empty values to 'other'
            payment_df.loc[payment_df['payment_type'].isna() & payment_df['pay_type'].notna() & (payment_df['pay_type_clean'] != 'NAN'), 'payment_type'] = 'other'
        
        # Get most frequent payment type per patient
        payment_counts = payment_df.groupby(['pat_id', 'payment_type']).size().reset_index(name='count')
        payment_counts = payment_counts.sort_values(['pat_id', 'count'], ascending=[True, False])
        payment_summary = payment_counts.groupby('pat_id').first().reset_index()[['pat_id', 'payment_type']]
        
        print(f"Unique patients with payment type: {len(payment_summary)}")
        print("\nPayment type distribution:")
        payment_dist = payment_summary['payment_type'].value_counts()
        for ptype, count in payment_dist.items():
            percentage = (count / len(payment_summary)) * 100
            print(f"  {ptype}: {count:,} ({percentage:.1f}%)")
    
    # Step 3: Merge all features
    print("\n3. Merging demographic features...")
    
    # Merge age/gender with payment type
    if not payment_summary.empty:
        final_demo_df = demo_df.merge(payment_summary, on='pat_id', how='outer')
    else:
        final_demo_df = demo_df.copy()
        final_demo_df['payment_type'] = np.nan
    
    # Save demographic features
    output_path = '/sharefolder/wanglab/MME/demographic_features_fixed.csv'
    final_demo_df.to_csv(output_path, index=False)
    print(f"\nSaved demographic features to: {output_path}")
    
    # Summary statistics
    print("\nFinal Summary:")
    print(f"Total unique patients: {len(final_demo_df)}")
    print(f"Patients with age: {final_demo_df['age'].notna().sum()}")
    print(f"Patients with gender: {final_demo_df['gender'].notna().sum()}")
    print(f"Patients with payment type: {final_demo_df['payment_type'].notna().sum()}")
    
    elapsed = time.time() - start_time
    print(f"\nExtraction completed in {elapsed/60:.1f} minutes")
    
    return final_demo_df

def merge_with_existing_dataset():
    """Merge the fixed demographic features with existing dataset"""
    print("\n" + "="*80)
    print("MERGING FIXED DEMOGRAPHIC FEATURES")
    print("="*80)
    
    # Check which dataset exists
    dataset_paths = [
        '/sharefolder/wanglab/MME/final_dataset_with_all_features.csv',
        '/sharefolder/wanglab/MME/final_dataset_with_oud_labels.csv'
    ]
    
    existing_path = None
    for path in dataset_paths:
        if os.path.exists(path):
            existing_path = path
            break
    
    if not existing_path:
        print("Error: No existing dataset found")
        return False
    
    demo_path = '/sharefolder/wanglab/MME/demographic_features_fixed.csv'
    
    if not os.path.exists(demo_path):
        print(f"Error: Demographic features not found: {demo_path}")
        return False
    
    try:
        print(f"Loading existing dataset from: {existing_path}")
        existing_df = pd.read_csv(existing_path)
        existing_df['pat_id'] = existing_df['pat_id'].astype(str)
        print(f"Existing dataset: {len(existing_df)} rows")
        
        # Check if we need to remove old demographic columns
        old_demo_cols = ['age', 'gender', 'payment_type'] + [col for col in existing_df.columns if col.startswith('payment_')]
        cols_to_drop = [col for col in old_demo_cols if col in existing_df.columns]
        
        if cols_to_drop:
            print(f"Removing old demographic columns: {cols_to_drop}")
            existing_df = existing_df.drop(columns=cols_to_drop)
        
        print("Loading fixed demographic features...")
        demo_df = pd.read_csv(demo_path)
        demo_df['pat_id'] = demo_df['pat_id'].astype(str)
        print(f"Demographic features: {len(demo_df)} patients")
        
        # Merge datasets
        print("Merging datasets...")
        final_df = existing_df.merge(demo_df, on='pat_id', how='left')
        
        # Create dummy variables for payment type
        if 'payment_type' in final_df.columns:
            # Fill NaN with 'unknown' for dummy encoding
            final_df['payment_type'] = final_df['payment_type'].fillna('unknown')
            
            # Create dummies without including NaN
            payment_dummies = pd.get_dummies(final_df['payment_type'], prefix='payment', dummy_na=False)
            
            # Remove payment_nan column if it exists
            if 'payment_nan' in payment_dummies.columns:
                payment_dummies = payment_dummies.drop('payment_nan', axis=1)
            
            final_df = pd.concat([final_df, payment_dummies], axis=1)
            
            # Remove the original payment_type column
            final_df = final_df.drop('payment_type', axis=1)
        
        # Save final dataset
        output_path = '/sharefolder/wanglab/MME/final_dataset_complete.csv'
        final_df.to_csv(output_path, index=False)
        print(f"\nSaved complete dataset to: {output_path}")
        
        # Print summary
        print("\nDataset Summary:")
        print(f"Total rows: {len(final_df)}")
        print(f"Total columns: {len(final_df.columns)}")
        
        # Feature completeness
        print("\nFeature completeness:")
        for col in ['age', 'gender']:
            if col in final_df.columns:
                completeness = (final_df[col].notna().sum() / len(final_df)) * 100
                print(f"  {col}: {completeness:.1f}%")
        
        # Payment type completeness
        payment_cols = [col for col in final_df.columns if col.startswith('payment_')]
        if payment_cols:
            # Check if any payment column is 1 (excluding payment_unknown)
            known_payment_cols = [col for col in payment_cols if not col.endswith('_unknown')]
            if known_payment_cols:
                has_known_payment = final_df[known_payment_cols].sum(axis=1) > 0
                payment_completeness = (has_known_payment.sum() / len(final_df)) * 100
                print(f"  payment_type (known): {payment_completeness:.1f}%")
                
                # Show distribution of payment types
                print("\nPayment type distribution in final dataset:")
                for col in sorted(payment_cols):
                    count = final_df[col].sum()
                    if count > 0:
                        percentage = (count / len(final_df)) * 100
                        print(f"  {col}: {count:,} ({percentage:.1f}%)")
        
        # List all feature columns
        print("\nAll features:")
        feature_cols = [col for col in final_df.columns 
                       if col not in ['pat_id', 'most_recent_date', 'first_oud_date', 'oud_year']]
        
        print(f"Total features: {len(feature_cols)}")
        
        return True
        
    except Exception as e:
        print(f"Error during merge: {e}")
        import traceback
        traceback.print_exc()
        return False

def main():
    """Main function"""
    print(" Starting Fixed Demographic Feature Extraction")
    print("Extracting: age (der_yob), gender (der_sex), payment type (pay_type)")
    
    # Extract demographic features
    demo_df = extract_demographic_features_fixed()
    
    if demo_df is not None and not demo_df.empty:
        # Merge with existing dataset
        if merge_with_existing_dataset():
            print("\n All demographic features successfully extracted and merged!")
            print("\nFinal dataset saved as: final_dataset_complete.csv")
            print("Use this dataset for your ML/NN training")
        else:
            print("\n Failed to merge demographic features")
    else:
        print("\n Failed to extract demographic features")

if __name__ == "__main__":
    main()