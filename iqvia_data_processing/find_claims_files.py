#!/usr/bin/env python3
"""
Debug script to check access to IQVIA claims files
"""

import os
import glob
import pandas as pd

def check_claims_access():
    """Check access to claims files with detailed debugging"""
    
    print("Debugging IQVIA claims file access...")
    print("="*80)
    
    year = '2006'
    
    # List of paths to check
    paths_to_check = [
        f'/sharefolder/IQVIA/claims_{year}/csv_in_parts',
        f'/sharefolder/IQVIA/claims_{year}',
        f'/sharefolder/IQVIA/{year}',
        f'/sharefolder/IQVIA',
    ]
    
    for path in paths_to_check:
        print(f"\nChecking: {path}")
        
        if os.path.exists(path):
            print(f"✓ Path exists")
            
            # Check if it's a directory
            if os.path.isdir(path):
                print(f"✓ Is a directory")
                
                # Try to list contents
                try:
                    contents = os.listdir(path)
                    print(f"✓ Can list contents ({len(contents)} items)")
                    
                    # Show first few items
                    dirs = [d for d in contents if os.path.isdir(os.path.join(path, d))]
                    files = [f for f in contents if os.path.isfile(os.path.join(path, f))]
                    
                    if dirs:
                        print(f"  Subdirectories ({len(dirs)}):")
                        for d in sorted(dirs)[:5]:
                            print(f"    📁 {d}")
                            # Check for CSV files in subdirectory
                            subdir_path = os.path.join(path, d)
                            csv_pattern = os.path.join(subdir_path, '*.csv')
                            csv_files = glob.glob(csv_pattern)
                            if csv_files:
                                print(f"       → Contains {len(csv_files)} CSV files")
                    
                    if files:
                        print(f"  Files ({len(files)}):")
                        csv_files = [f for f in files if f.endswith('.csv')]
                        other_files = [f for f in files if not f.endswith('.csv')]
                        
                        if csv_files:
                            print(f"    CSV files ({len(csv_files)}):")
                            for f in sorted(csv_files)[:5]:
                                print(f"      📄 {f}")
                                # Check file size
                                size = os.path.getsize(os.path.join(path, f))
                                print(f"         Size: {size/1024/1024:.1f} MB")
                        
                        if other_files:
                            print(f"    Other files ({len(other_files)}):")
                            for f in sorted(other_files)[:5]:
                                print(f"      📄 {f}")
                    
                    # Look for CSV files recursively
                    print(f"\n  Searching for CSV files recursively...")
                    csv_pattern = os.path.join(path, '**', '*.csv')
                    all_csv_files = glob.glob(csv_pattern, recursive=True)
                    if all_csv_files:
                        print(f"  ✓ Found {len(all_csv_files)} CSV files total")
                        print(f"  Sample paths:")
                        for f in all_csv_files[:3]:
                            rel_path = os.path.relpath(f, path)
                            print(f"    - {rel_path}")
                    
                except PermissionError as e:
                    print(f"✗ Permission denied when listing contents: {e}")
                except Exception as e:
                    print(f"✗ Error listing contents: {e}")
            else:
                print(f"✗ Not a directory")
        else:
            print(f"✗ Path does not exist")
    
    # Try to read a sample file if found
    print("\n" + "-"*80)
    print("Attempting to read a sample claims file...")
    
    # Look for CSV files in the most likely location
    sample_path = f'/sharefolder/IQVIA/claims_{year}'
    if os.path.exists(sample_path):
        # Find CSV files
        csv_files = []
        for root, dirs, files in os.walk(sample_path):
            for file in files:
                if file.endswith('.csv') and 'part' in file.lower():
                    csv_files.append(os.path.join(root, file))
        
        if csv_files:
            sample_file = csv_files[0]
            print(f"\nTrying to read: {sample_file}")
            try:
                # Read just first few lines
                df = pd.read_csv(sample_file, sep='|', nrows=5, header=None)
                print(f"✓ Successfully read file!")
                print(f"  Shape: {df.shape}")
                print(f"  First row preview:")
                print(f"  {df.iloc[0].values[:10]}...")
            except Exception as e:
                print(f"✗ Error reading file: {e}")

if __name__ == "__main__":
    check_claims_access()