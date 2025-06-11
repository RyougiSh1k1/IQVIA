"""
This script merges MME features and prescriber features into a final dataset.
Since we don't have ICD data yet, it creates dummy labels.

Input files:
- MME_features_200_max.csv (MME-related features)
- prscbr_features.csv (prescriber-related features)

Output file:
- final_features.csv (merged feature dataset ready for modeling)
"""

import pandas as pd
import time
import os

def merge_features():
    """
    Merge MME and prescriber features into final dataset
    """
    start_time = time.time()
    
    print("Starting feature merge process...")
    
    # Define file paths
    mme_path = '/sharefolder/wanglab/MME/MME_features_200_max.csv'
    prescriber_path = '/sharefolder/wanglab/MME/prscbr_features.csv'
    output_path = '/sharefolder/wanglab/MME/final_features.csv'
    
    # Check if files exist
    if not os.path.exists(mme_path):
        print("Error: MME features file not found: {}".format(mme_path))
        return False
        
    if not os.path.exists(prescriber_path):
        print("Error: Prescriber features file not found: {}".format(prescriber_path))
        return False
    
    try:
        # Load feature files
        print("Loading MME features...")
        mme_df = pd.read_csv(mme_path)
        print("MME features loaded: {} rows, {} columns".format(len(mme_df), len(mme_df.columns)))
        print("MME columns: {}".format(list(mme_df.columns)))
        
        print("\nLoading prescriber features...")
        prescriber_df = pd.read_csv(prescriber_path)
        print("Prescriber features loaded: {} rows, {} columns".format(len(prescriber_df), len(prescriber_df.columns)))
        print("Prescriber columns: {}".format(list(prescriber_df.columns)))
        
        # Check for common patients
        common_patients = set(mme_df['pat_id']).intersection(set(prescriber_df['pat_id']))
        print("\nCommon patients between datasets: {}".format(len(common_patients)))
        
        # Merge on patient ID and most recent date
        print("\nMerging datasets on pat_id and most_recent_date...")
        merged_df = pd.merge(
            mme_df, 
            prescriber_df, 
            on=['pat_id', 'most_recent_date'], 
            how='inner'
        )
        
        print("Merged dataset shape: {} rows, {} columns".format(len(merged_df), len(merged_df.columns)))
        print("Merged columns: {}".format(list(merged_df.columns)))
        
        # Check for duplicates
        duplicates = merged_df.duplicated(subset=['pat_id', 'most_recent_date']).sum()
        if duplicates > 0:
            print("Warning: {} duplicate entries found. Removing duplicates...".format(duplicates))
            merged_df = merged_df.drop_duplicates(subset=['pat_id', 'most_recent_date'])
            print("Dataset after removing duplicates: {} rows".format(len(merged_df)))
        
        # Display summary statistics
        print("\nFeature Summary Statistics:")
        feature_cols = ['MME_last_365_days', 'MME_last_2_years', 'MME_prior_1_year', 
                       'MME_120_2_years', 'prscbr_last_2_years', 'prscrbr_last_180_days']
        
        for col in feature_cols:
            if col in merged_df.columns:
                print("\n{}:".format(col))
                print("  Mean: {:.2f}".format(merged_df[col].mean()))
                print("  Median: {:.2f}".format(merged_df[col].median()))
                print("  Std: {:.2f}".format(merged_df[col].std()))
                print("  Min: {:.2f}".format(merged_df[col].min()))
                print("  Max: {:.2f}".format(merged_df[col].max()))
                print("  Missing: {}".format(merged_df[col].isnull().sum()))
        
        # Save merged dataset
        merged_df.to_csv(output_path, index=False)
        print("\nFinal features saved to: {}".format(output_path))
        print("Total patients in final dataset: {}".format(len(merged_df)))
        
        # Create a version with dummy labels for testing
        print("\nCreating dataset with dummy labels for testing...")
        merged_df['oud_label'] = 0  # All patients labeled as no OUD
        merged_df['num_oud_events'] = 0
        merged_df['first_oud_date'] = None
        merged_df['latest_oud_date'] = None
        
        dummy_output_path = '/sharefolder/wanglab/MME/final_dataset_with_labels.csv'
        merged_df.to_csv(dummy_output_path, index=False)
        print("Dataset with dummy labels saved to: {}".format(dummy_output_path))
        print("Note: Replace with actual ICD-based labels when available")
        
        end_time = time.time()
        elapsed_time = end_time - start_time
        print("\nElapsed Time: {:.2f} seconds".format(elapsed_time))
        
        return True
        
    except Exception as e:
        print("Error during merge: {}".format(str(e)))
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = merge_features()
    if success:
        print("\nFeature merge completed successfully!")
        print("Next steps:")
        print("1. Run 6_extract_OUD_labels.py when ICD data is available")
        print("2. Or run simple_test.py to test ML models with dummy labels")
    else:
        print("\nFeature merge failed!")
        exit(1)