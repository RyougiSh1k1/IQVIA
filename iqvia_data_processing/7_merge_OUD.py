"""
Step 7: Merge Features with OUD Labels from All Years
This script merges the feature dataset with OUD labels extracted from all years (2006-2022)

Input files:
- final_features.csv (from step 5: merged MME and prescriber features)
- oud_patients_all_years.csv (from step 6: patients with OUD from all years)

Output files:
- final_dataset_with_oud_labels.csv (ready for ML modeling)
- dataset_statistics.txt (summary statistics)
"""

import pandas as pd
import numpy as np
import time
import os
from datetime import datetime

def create_patient_labels(features_df, oud_df):
    """
    Create comprehensive patient labels dataset
    """
    print("\nCreating patient labels...")
    
    # Get all unique patients from features
    all_patients = set(features_df['pat_id'].astype(str).unique())
    print(f"Total patients in features: {len(all_patients):,}")
    
    # Get OUD patients
    oud_patients = set(oud_df['pat_id'].astype(str).unique())
    print(f"Total OUD patients: {len(oud_patients):,}")
    
    # Create labels DataFrame
    labels_data = []
    
    for pat_id in all_patients:
        if pat_id in oud_patients:
            # Get OUD info
            oud_info = oud_df[oud_df['pat_id'].astype(str) == pat_id].iloc[0]
            labels_data.append({
                'pat_id': pat_id,
                'oud_label': 1,
                'first_oud_date': oud_info.get('service_date', ''),
                'matched_icd_codes': oud_info.get('matched_icd_codes', ''),
                'oud_year': oud_info.get('year', '')
            })
        else:
            labels_data.append({
                'pat_id': pat_id,
                'oud_label': 0,
                'first_oud_date': None,
                'matched_icd_codes': '',
                'oud_year': ''
            })
    
    labels_df = pd.DataFrame(labels_data)
    return labels_df

def merge_features_with_oud_labels():
    """
    Merge feature dataset with OUD labels based on patient IDs
    """
    start_time = time.time()
    
    print("="*80)
    print("STEP 7: MERGING FEATURES WITH OUD LABELS (ALL YEARS)")
    print("="*80)
    
    # Define file paths
    features_path = '/sharefolder/wanglab/MME/final_features.csv'
    oud_labels_path = '/sharefolder/wanglab/MME/oud_patients_all_years.csv'
    output_path = '/sharefolder/wanglab/MME/final_dataset_with_oud_labels.csv'
    stats_path = '/sharefolder/wanglab/MME/dataset_statistics.txt'
    
    # Check if files exist
    if not os.path.exists(features_path):
        print(f"Error: Features file not found: {features_path}")
        return False
        
    if not os.path.exists(oud_labels_path):
        print(f"Error: OUD labels file not found: {oud_labels_path}")
        print("Please ensure step 6 (extract_OUD_labels_all_years.py) has been run successfully")
        return False
    
    try:
        # Load the features dataset
        print("\nLoading features dataset...")
        features_df = pd.read_csv(features_path)
        print(f"Features dataset loaded: {len(features_df):,} rows, {len(features_df.columns)} columns")
        
        # Ensure pat_id is string type for consistent merging
        features_df['pat_id'] = features_df['pat_id'].astype(str)
        
        # Load OUD labels
        print("\nLoading OUD labels from all years...")
        oud_df = pd.read_csv(oud_labels_path)
        oud_df['pat_id'] = oud_df['pat_id'].astype(str)
        print(f"OUD patients loaded: {len(oud_df):,} unique patients")
        
        # Create patient labels
        labels_df = create_patient_labels(features_df, oud_df)
        
        # Merge features with labels
        print("\nMerging features with OUD labels...")
        final_df = features_df.merge(
            labels_df[['pat_id', 'oud_label', 'first_oud_date', 'oud_year']], 
            on='pat_id', 
            how='left'
        )
        
        # Fill any missing OUD labels with 0
        final_df['oud_label'] = final_df['oud_label'].fillna(0).astype(int)
        
        # Calculate statistics
        total_count = len(final_df)
        oud_count = final_df['oud_label'].sum()
        non_oud_count = total_count - oud_count
        prevalence = (oud_count / total_count * 100) if total_count > 0 else 0
        
        # Year-wise OUD distribution
        year_stats = None
        if 'oud_year' in final_df.columns:
            year_stats = final_df[final_df['oud_label'] == 1]['oud_year'].value_counts().sort_index()
        
        # Save the merged dataset
        print("\nSaving merged dataset...")
        final_df.to_csv(output_path, index=False)
        print(f"✓ Merged dataset saved to: {output_path}")
        
        # Generate comprehensive statistics
        print("\n" + "="*50)
        print("DATASET STATISTICS")
        print("="*50)
        print(f"Total patients: {total_count:,}")
        print(f"Patients with OUD: {oud_count:,}")
        print(f"Patients without OUD: {non_oud_count:,}")
        print(f"OUD prevalence: {prevalence:.2f}%")
        print(f"Features: {len(final_df.columns)} columns")
        
        # Feature statistics
        feature_cols = [col for col in final_df.columns 
                       if col not in ['pat_id', 'oud_label', 'first_oud_date', 'oud_year']]
        
        print(f"\nFeature columns ({len(feature_cols)}):")
        for i, col in enumerate(feature_cols[:10]):  # Show first 10
            print(f"  {i+1}. {col}")
        if len(feature_cols) > 10:
            print(f"  ... and {len(feature_cols) - 10} more")
        
        # Save detailed statistics
        with open(stats_path, 'w') as f:
            f.write("IQVIA OUD DATASET STATISTICS\n")
            f.write("="*50 + "\n")
            f.write(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write(f"Processing time: {(time.time() - start_time)/60:.1f} minutes\n\n")
            
            f.write("DATASET OVERVIEW\n")
            f.write("-"*30 + "\n")
            f.write(f"Total patients: {total_count:,}\n")
            f.write(f"Patients with OUD: {oud_count:,}\n")
            f.write(f"Patients without OUD: {non_oud_count:,}\n")
            f.write(f"OUD prevalence: {prevalence:.2f}%\n")
            f.write(f"Total features: {len(feature_cols)}\n\n")
            
            if year_stats is not None and len(year_stats) > 0:
                f.write("OUD CASES BY YEAR\n")
                f.write("-"*30 + "\n")
                for year, count in year_stats.items():
                    f.write(f"{year}: {count:,}\n")
                f.write("\n")
            
            f.write("FEATURE COLUMNS\n")
            f.write("-"*30 + "\n")
            for col in feature_cols:
                f.write(f"- {col}\n")
            
            f.write("\nFEATURE STATISTICS\n")
            f.write("-"*30 + "\n")
            
            # Calculate basic statistics for numeric features
            numeric_features = final_df[feature_cols].select_dtypes(include=[np.number])
            if not numeric_features.empty:
                stats_summary = numeric_features.describe().round(2)
                f.write(stats_summary.to_string())
        
        print(f"\n✓ Statistics saved to: {stats_path}")
        
        # Create stratified samples for validation
        print("\nCreating stratified samples...")
        
        # Small sample (1,000 rows)
        create_stratified_sample(final_df, 1000, prevalence, 
                               '/sharefolder/wanglab/MME/final_dataset_sample_1000.csv')
        
        # Medium sample (10,000 rows)
        if len(final_df) >= 10000:
            create_stratified_sample(final_df, 10000, prevalence, 
                                   '/sharefolder/wanglab/MME/final_dataset_sample_10000.csv')
        
        # Large sample (100,000 rows)
        if len(final_df) >= 100000:
            create_stratified_sample(final_df, 100000, prevalence, 
                                   '/sharefolder/wanglab/MME/final_dataset_sample_100000.csv')
        
        print("\n" + "="*50)
        print("NEXT STEPS")
        print("="*50)
        print("1. Review the dataset statistics above")
        print("2. Check the stratified samples for data quality")
        print("3. Run ML models using final_dataset_with_oud_labels.csv")
        print("4. Consider addressing class imbalance if needed")
        print("\nDataset is ready for ML modeling!")
        
        return True
        
    except Exception as e:
        print(f"\nError during merge: {str(e)}")
        import traceback
        traceback.print_exc()
        return False

def create_stratified_sample(df, sample_size, prevalence, output_path):
    """Create a stratified sample maintaining OUD prevalence"""
    try:
        if len(df) < sample_size:
            print(f"Dataset smaller than requested sample size {sample_size}")
            return
        
        # Calculate stratified sizes
        oud_sample_size = int(sample_size * prevalence / 100)
        non_oud_sample_size = sample_size - oud_sample_size
        
        # Get OUD and non-OUD patients
        oud_patients = df[df['oud_label'] == 1]
        non_oud_patients = df[df['oud_label'] == 0]
        
        # Sample from each group
        oud_sample = oud_patients.sample(
            n=min(oud_sample_size, len(oud_patients)), 
            random_state=42
        )
        non_oud_sample = non_oud_patients.sample(
            n=min(non_oud_sample_size, len(non_oud_patients)), 
            random_state=42
        )
        
        # Combine and shuffle
        stratified_sample = pd.concat([oud_sample, non_oud_sample])
        stratified_sample = stratified_sample.sample(frac=1, random_state=42)
        
        # Save
        stratified_sample.to_csv(output_path, index=False)
        print(f"✓ Stratified sample ({sample_size:,} rows) saved to: {os.path.basename(output_path)}")
        
    except Exception as e:
        print(f"Error creating stratified sample: {e}")

def validate_merged_dataset():
    """
    Validate the merged dataset to ensure it's ready for ML modeling
    """
    print("\n" + "="*80)
    print("VALIDATING MERGED DATASET")
    print("="*80)
    
    output_path = '/sharefolder/wanglab/MME/final_dataset_with_oud_labels.csv'
    
    if not os.path.exists(output_path):
        print("Error: Merged dataset not found. Please run merge_features_with_oud_labels() first.")
        return False
    
    try:
        # Load dataset
        print("Loading dataset for validation...")
        df = pd.read_csv(output_path, nrows=10000)  # Load sample for validation
        
        # Check for required columns
        required_cols = ['pat_id', 'oud_label']
        missing_cols = [col for col in required_cols if col not in df.columns]
        
        if missing_cols:
            print(f"❌ Missing required columns: {missing_cols}")
            return False
        
        # Check data types
        print("\nData type validation:")
        print(f"  - pat_id type: {df['pat_id'].dtype}")
        print(f"  - oud_label type: {df['oud_label'].dtype}")
        print(f"  - oud_label unique values: {sorted(df['oud_label'].unique())}")
        
        # Check for missing values
        print("\nMissing value check:")
        missing_counts = df.isnull().sum()
        cols_with_missing = missing_counts[missing_counts > 0]
        
        if len(cols_with_missing) > 0:
            print("  Columns with missing values:")
            for col, count in cols_with_missing.items():
                pct = (count / len(df)) * 100
                print(f"    - {col}: {count} ({pct:.1f}%)")
        else:
            print("  ✓ No missing values in key columns")
        
        # Check feature distributions
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        numeric_cols = [col for col in numeric_cols if col != 'oud_label']
        
        print(f"\n✓ Dataset validation complete")
        print(f"  - {len(numeric_cols)} numeric features found")
        print(f"  - Dataset appears ready for ML modeling")
        
        return True
        
    except Exception as e:
        print(f"Error during validation: {e}")
        return False

def main():
    """Main function"""
    print("🚀 Starting Step 7: Merge Features with OUD Labels")
    
    # Run the merge
    if merge_features_with_oud_labels():
        print("\n Merge completed successfully!")
        
        # Validate the result
        validate_merged_dataset()
    else:
        print("\n Merge failed! Please check the error messages above.")

if __name__ == "__main__":
    main()