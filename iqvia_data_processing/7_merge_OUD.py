"""
Step 7: Merge Features with OUD Labels
This script merges the feature dataset from step 5 with OUD labels from step 6

Input files:
- final_features.csv (from step 5: merged MME and prescriber features)
- oud_patients_2006.csv (from step 6: patients with OUD labels)

Output file:
- final_dataset_with_oud_labels.csv (ready for ML modeling)
"""

import pandas as pd
import numpy as np
import time
import os

def merge_features_with_oud_labels():
    """
    Merge feature dataset with OUD labels based on patient IDs
    """
    start_time = time.time()
    
    print("="*80)
    print("STEP 7: MERGING FEATURES WITH OUD LABELS")
    print("="*80)
    
    # Define file paths
    features_path = '/sharefolder/wanglab/MME/final_features.csv'
    oud_labels_path = '/sharefolder/wanglab/MME/oud_patients_2006.csv'
    output_path = '/sharefolder/wanglab/MME/final_dataset_with_oud_labels.csv'
    
    # Check if files exist
    if not os.path.exists(features_path):
        print(f"Error: Features file not found: {features_path}")
        return False
        
    if not os.path.exists(oud_labels_path):
        print(f"Error: OUD labels file not found: {oud_labels_path}")
        print("Please ensure step 6 (extract_OUD_labels.py) has been run successfully")
        return False
    
    try:
        # Load the features dataset
        print("\nLoading features dataset...")
        features_df = pd.read_csv(features_path)
        print(f"Features dataset loaded: {len(features_df):,} rows, {len(features_df.columns)} columns")
        print(f"Features columns: {list(features_df.columns)}")
        
        # Ensure pat_id is string type for consistent merging
        features_df['pat_id'] = features_df['pat_id'].astype(str)
        
        # Load OUD labels
        print("\nLoading OUD labels...")
        oud_df = pd.read_csv(oud_labels_path)
        print(f"OUD patients loaded: {len(oud_df):,} patients identified with OUD")
        
        # Ensure pat_id is string type
        oud_df['pat_id'] = oud_df['pat_id'].astype(str)
        
        # Get unique OUD patients
        oud_patients = set(oud_df['pat_id'].unique())
        print(f"Unique OUD patients: {len(oud_patients):,}")
        
        # Create OUD label column (1 for OUD patients, 0 for others)
        print("\nCreating OUD labels for all patients...")
        features_df['oud_label'] = features_df['pat_id'].apply(
            lambda x: 1 if x in oud_patients else 0
        )
        
        # Add additional OUD information if available
        if 'matched_icd_codes' in oud_df.columns:
            # Create a dictionary of patient to ICD codes
            patient_icd_dict = dict(zip(oud_df['pat_id'], oud_df['matched_icd_codes']))
            features_df['oud_icd_codes'] = features_df['pat_id'].apply(
                lambda x: patient_icd_dict.get(x, '')
            )
        
        if 'service_date' in oud_df.columns:
            # Get first OUD diagnosis date for each patient
            oud_first_date = oud_df.groupby('pat_id')['service_date'].min().to_dict()
            features_df['first_oud_date'] = features_df['pat_id'].apply(
                lambda x: oud_first_date.get(x, '')
            )
        
        # Calculate OUD prevalence
        oud_count = features_df['oud_label'].sum()
        total_count = len(features_df)
        prevalence = (oud_count / total_count) * 100
        
        print("\n" + "-"*50)
        print("DATASET SUMMARY")
        print("-"*50)
        print(f"Total patients: {total_count:,}")
        print(f"OUD patients: {oud_count:,}")
        print(f"Non-OUD patients: {total_count - oud_count:,}")
        print(f"OUD prevalence: {prevalence:.2f}%")
        print(f"Class imbalance ratio: 1:{(total_count - oud_count) / max(oud_count, 1):.1f}")
        
        # Feature statistics by OUD status
        print("\n" + "-"*50)
        print("FEATURE STATISTICS BY OUD STATUS")
        print("-"*50)
        
        feature_cols = ['MME_last_365_days', 'MME_last_2_years', 'MME_prior_1_year', 
                       'MME_120_2_years', 'prscbr_last_2_years', 'prscrbr_last_180_days']
        
        for col in feature_cols:
            if col in features_df.columns:
                oud_mean = features_df[features_df['oud_label'] == 1][col].mean()
                non_oud_mean = features_df[features_df['oud_label'] == 0][col].mean()
                print(f"\n{col}:")
                print(f"  OUD patients mean: {oud_mean:.2f}")
                print(f"  Non-OUD patients mean: {non_oud_mean:.2f}")
                print(f"  Ratio (OUD/Non-OUD): {oud_mean/max(non_oud_mean, 0.001):.2f}x")
        
        # Check for any missing values
        print("\n" + "-"*50)
        print("DATA QUALITY CHECK")
        print("-"*50)
        missing_counts = features_df[feature_cols + ['oud_label']].isnull().sum()
        if missing_counts.sum() == 0:
            print("✓ No missing values in features or labels")
        else:
            print("Missing values found:")
            for col, count in missing_counts[missing_counts > 0].items():
                print(f"  {col}: {count} ({count/len(features_df)*100:.2f}%)")
        
        # Save the final dataset
        print("\n" + "-"*50)
        print("SAVING FINAL DATASET")
        print("-"*50)
        features_df.to_csv(output_path, index=False)
        print(f"✓ Final dataset saved to: {output_path}")
        print(f"  Dimensions: {len(features_df):,} rows × {len(features_df.columns)} columns")
        print(f"  File size: {os.path.getsize(output_path) / (1024*1024):.1f} MB")
        
        # Create a stratified sample for quick validation
        print("\nCreating stratified sample for validation...")
        sample_size = min(1000, len(features_df))
        
        # Stratified sampling to maintain OUD prevalence
        oud_sample_size = int(sample_size * prevalence / 100)
        non_oud_sample_size = sample_size - oud_sample_size
        
        oud_sample = features_df[features_df['oud_label'] == 1].sample(
            n=min(oud_sample_size, oud_count), random_state=42
        )
        non_oud_sample = features_df[features_df['oud_label'] == 0].sample(
            n=min(non_oud_sample_size, total_count - oud_count), random_state=42
        )
        
        stratified_sample = pd.concat([oud_sample, non_oud_sample]).sample(frac=1, random_state=42)
        sample_path = '/sharefolder/wanglab/MME/final_dataset_sample_1000.csv'
        stratified_sample.to_csv(sample_path, index=False)
        print(f"✓ Stratified sample saved to: {sample_path}")
        
        end_time = time.time()
        elapsed_time = end_time - start_time
        print(f"\n✓ Step 7 completed successfully in {elapsed_time:.2f} seconds")
        
        return True
        
    except Exception as e:
        print(f"\nError during merge: {str(e)}")
        import traceback
        traceback.print_exc()
        return False

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
        # Load the dataset
        df = pd.read_csv(output_path)
        print(f"Dataset loaded: {len(df):,} rows")
        
        # Required columns for ML
        required_features = [
            'pat_id', 'most_recent_date',
            'MME_last_365_days', 'MME_last_2_years', 'MME_prior_1_year',
            'MME_120_2_years', 'prscbr_last_2_years', 'prscrbr_last_180_days',
            'oud_label'
        ]
        
        missing_cols = [col for col in required_features if col not in df.columns]
        if missing_cols:
            print(f"❌ Missing required columns: {missing_cols}")
            return False
        else:
            print("✓ All required columns present")
        
        # Check data types
        print("\nData types:")
        for col in required_features[2:]:  # Skip pat_id and date
            dtype = df[col].dtype
            print(f"  {col}: {dtype}")
            if col == 'oud_label' and dtype not in ['int64', 'int32']:
                print(f"    ⚠️  Warning: oud_label should be integer type")
        
        # Check value ranges
        print("\nValue range validation:")
        
        # OUD label should be binary
        unique_labels = df['oud_label'].unique()
        if set(unique_labels) == {0, 1}:
            print("✓ OUD labels are binary (0, 1)")
        else:
            print(f"❌ Invalid OUD label values: {unique_labels}")
        
        # Features should be non-negative
        feature_cols = required_features[2:-1]  # Exclude pat_id, date, and label
        for col in feature_cols:
            min_val = df[col].min()
            if min_val < 0:
                print(f"❌ {col} has negative values (min: {min_val})")
            else:
                print(f"✓ {col} values are non-negative")
        
        # Check for duplicates
        duplicates = df.duplicated(subset=['pat_id', 'most_recent_date']).sum()
        if duplicates > 0:
            print(f"\n⚠️  Warning: {duplicates} duplicate patient records found")
        else:
            print("\n✓ No duplicate patient records")
        
        print("\n✓ Dataset validation complete - ready for ML modeling!")
        return True
        
    except Exception as e:
        print(f"Error during validation: {str(e)}")
        return False

if __name__ == "__main__":
    print("Starting Step 7: Merge Features with OUD Labels")
    print("This step combines the feature dataset with OUD diagnosis labels\n")
    
    # Run the merge
    success = merge_features_with_oud_labels()
    
    if success:
        print("\n" + "="*80)
        # Validate the result
        validate_merged_dataset()
        
        print("\n" + "="*80)
        print("NEXT STEPS")
        print("="*80)
        print("1. Review the final dataset statistics above")
        print("2. Check the stratified sample for data quality")
        print("3. Run ML models using final_dataset_with_oud_labels.csv")
        print("4. Consider addressing class imbalance if OUD prevalence is very low")
        print("\nYou can now run simple_test.py or your ML pipeline with real OUD labels!")
    else:
        print("\nMerge failed! Please check the error messages above.")
        print("Ensure that:")
        print("- Step 5 (merge_features.py) completed successfully")
        print("- Step 6 (extract_OUD_labels.py) completed successfully")
        print("- Both output files exist in /sharefolder/wanglab/MME/")