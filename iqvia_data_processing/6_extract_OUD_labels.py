"""
Since we cannot access the raw claims files with ICD codes,
this script creates a template for OUD labels using the filtered prescription data.

This creates a dataset with all unique patients from the opioid prescriptions,
ready to be merged with OUD labels when ICD data becomes available.
"""

import pandas as pd
import os
import time

def create_patient_list_from_filtered_data(year='2006'):
    """Extract unique patients from filtered prescription data"""
    
    print(f"Processing filtered data for year {year}...")
    
    # Read the filtered prescription data
    filtered_file = f'/sharefolder/wanglab/iqvia_ndc_{year}.csv'
    
    if not os.path.exists(filtered_file):
        print(f"ERROR: Filtered file not found: {filtered_file}")
        return None
    
    print(f"Reading filtered prescriptions from: {filtered_file}")
    
    # Read only patient IDs to save memory
    df = pd.read_csv(filtered_file, usecols=['pat_id'])
    
    # Get unique patients
    unique_patients = df['pat_id'].unique()
    print(f"Found {len(unique_patients)} unique patients with opioid prescriptions")
    
    # Create patient list dataframe
    patient_df = pd.DataFrame({
        'pat_id': unique_patients,
        'has_opioid_prescription': 1,
        'oud_label': 0,  # Default to 0, will need to be updated with ICD data
        'note': 'OUD label needs to be determined from ICD codes in claims data'
    })
    
    # Save patient list
    output_path = f'/sharefolder/wanglab/MME/patient_list_{year}.csv'
    patient_df.to_csv(output_path, index=False)
    print(f"Saved patient list to: {output_path}")
    
    return patient_df

def merge_with_features(year='2006'):
    """Merge patient list with existing features"""
    
    features_path = '/sharefolder/wanglab/MME/final_features.csv'
    patient_list_path = f'/sharefolder/wanglab/MME/patient_list_{year}.csv'
    
    if os.path.exists(features_path) and os.path.exists(patient_list_path):
        print("\nMerging with feature dataset...")
        
        features_df = pd.read_csv(features_path)
        patient_df = pd.read_csv(patient_list_path)
        
        # Add OUD label column (all 0 for now)
        features_df['oud_label'] = 0
        features_df['oud_data_available'] = 'No - ICD codes not accessible'
        
        # Save
        output_path = f'/sharefolder/wanglab/MME/final_dataset_template_{year}.csv'
        features_df.to_csv(output_path, index=False)
        
        print(f"Created template dataset: {output_path}")
        print(f"Total patients: {len(features_df)}")
        print("\nNOTE: OUD labels are all set to 0. You need to:")
        print("1. Get access to claims files with ICD codes")
        print("2. Run the full OUD extraction script")
        print("3. Or manually match ICD codes if you have them from another source")
        
        return features_df
    
    return None

def main():
    """Main function"""
    start_time = time.time()
    
    print("="*80)
    print("CREATING OUD LABEL TEMPLATE FROM FILTERED DATA")
    print("="*80)
    print("\nNOTE: This creates a template only, since we cannot access ICD codes")
    print("from the filtered prescription files.")
    
    year = '2006'
    
    # Create patient list
    patient_df = create_patient_list_from_filtered_data(year)
    
    if patient_df is not None:
        # Try to merge with features
        merge_with_features(year)
    
    # Provide instructions
    print("\n" + "="*80)
    print("NEXT STEPS:")
    print("="*80)
    print("To get actual OUD labels, you need to either:")
    print("\n1. Get read access to /sharefolder/IQVIA/claims_2006/csv_in_parts/")
    print("   - Contact your system administrator")
    print("   - Or run the extraction script from a location with proper permissions")
    print("\n2. Extract ICD codes separately:")
    print("   - Create a script that only extracts pat_id and diag1-5 columns")
    print("   - Run it with appropriate permissions")
    print("   - Match against your ICD code list")
    print("\n3. Use an alternative data source for OUD diagnoses")
    
    elapsed = time.time() - start_time
    print(f"\nTotal time: {elapsed:.1f} seconds")

if __name__ == "__main__":
    main()