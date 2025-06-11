"""
Extract sample data from merged features file for review
"""

import pandas as pd
import numpy as np

def extract_samples():
    """Extract various samples from the merged features dataset"""
    
    print("Loading merged features dataset...")
    
    # Load the data
    df = pd.read_csv('/sharefolder/wanglab/MME/final_features.csv')
    print(f"Total records: {len(df):,}")
    
    # Set random seed for reproducibility
    np.random.seed(42)
    
    # 1. Random sample of 100 records
    random_sample = df.sample(n=100, random_state=42)
    random_sample.to_csv('sample_random_100.csv', index=False)
    print("✓ Saved random sample of 100 records to 'sample_random_100.csv'")
    
    # 2. First 1000 records
    first_1000 = df.head(1000)
    first_1000.to_csv('sample_first_1000.csv', index=False)
    print("✓ Saved first 1000 records to 'sample_first_1000.csv'")
    
    # 3. Stratified sample based on MME levels
    # Create MME categories for stratification
    df['MME_category'] = pd.cut(df['MME_last_365_days'], 
                                bins=[0, 50, 90, 120, float('inf')],
                                labels=['Low', 'Medium', 'High', 'Very High'])
    
    # Take 25 samples from each category
    stratified_samples = []
    for category in ['Low', 'Medium', 'High', 'Very High']:
        category_data = df[df['MME_category'] == category]
        if len(category_data) >= 25:
            sample = category_data.sample(n=25, random_state=42)
        else:
            sample = category_data  # Take all if less than 25
        stratified_samples.append(sample)
    
    stratified_sample = pd.concat(stratified_samples)
    stratified_sample = stratified_sample.drop('MME_category', axis=1)  # Remove temporary column
    stratified_sample.to_csv('sample_stratified_mme.csv', index=False)
    print("✓ Saved stratified sample by MME levels to 'sample_stratified_mme.csv'")
    
    # 4. High-risk patients sample (multiple criteria)
    high_risk = df[
        (df['MME_last_365_days'] > 90) & 
        (df['prscbr_last_2_years'] > 2) &
        (df['MME_120_2_years'] > 0)
    ]
    high_risk_sample = high_risk.head(50) if len(high_risk) >= 50 else high_risk
    high_risk_sample.to_csv('sample_high_risk_patients.csv', index=False)
    print(f"✓ Saved {len(high_risk_sample)} high-risk patient samples to 'sample_high_risk_patients.csv'")
    
    # 5. Summary statistics
    summary_stats = df.describe()
    summary_stats.to_csv('feature_summary_stats.csv')
    print("✓ Saved summary statistics to 'feature_summary_stats.csv'")
    
    # 6. Sample with specific interesting patterns
    interesting_patterns = []
    
    # Patients with high MME but single prescriber
    pattern1 = df[(df['MME_last_365_days'] > 120) & (df['prscbr_last_2_years'] == 1)].head(10)
    if len(pattern1) > 0:
        pattern1['pattern'] = 'High MME, Single Prescriber'
        interesting_patterns.append(pattern1)
    
    # Patients with many prescribers but low MME
    pattern2 = df[(df['MME_last_365_days'] < 50) & (df['prscbr_last_2_years'] > 3)].head(10)
    if len(pattern2) > 0:
        pattern2['pattern'] = 'Low MME, Many Prescribers'
        interesting_patterns.append(pattern2)
    
    # Patients with prior year MME but no recent MME
    pattern3 = df[(df['MME_prior_1_year'] > 100) & (df['MME_last_365_days'] < 10)].head(10)
    if len(pattern3) > 0:
        pattern3['pattern'] = 'High Prior MME, Low Recent MME'
        interesting_patterns.append(pattern3)
    
    if interesting_patterns:
        interesting_sample = pd.concat(interesting_patterns)
        interesting_sample.to_csv('sample_interesting_patterns.csv', index=False)
        print(f"✓ Saved {len(interesting_sample)} interesting pattern samples to 'sample_interesting_patterns.csv'")
    
    print("\n📊 Dataset Overview:")
    print(f"Total patients: {len(df):,}")
    print(f"Date range: {df['most_recent_date'].min()} to {df['most_recent_date'].max()}")
    print(f"\nFeature ranges:")
    print(f"  MME_last_365_days: {df['MME_last_365_days'].min():.2f} - {df['MME_last_365_days'].max():.2f}")
    print(f"  MME_last_2_years: {df['MME_last_2_years'].min():.2f} - {df['MME_last_2_years'].max():.2f}")
    print(f"  prscbr_last_2_years: {df['prscbr_last_2_years'].min():.0f} - {df['prscbr_last_2_years'].max():.0f}")
    print(f"  prscrbr_last_180_days: {df['prscrbr_last_180_days'].min():.0f} - {df['prscrbr_last_180_days'].max():.0f}")

if __name__ == "__main__":
    extract_samples()