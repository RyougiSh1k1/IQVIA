#!/usr/bin/env python3
"""
Find where the claims CSV files are located
"""

import os
import glob

def find_claims_files():
    """Search for claims CSV files in various locations"""
    
    print("Searching for IQVIA claims files...")
    print("="*60)
    
    # Check IQVIA base directory
    base_dir = '/sharefolder/IQVIA'
    if os.path.exists(base_dir):
        print(f"\nContents of {base_dir}:")
        try:
            items = sorted(os.listdir(base_dir))
            for item in items:
                item_path = os.path.join(base_dir, item)
                if os.path.isdir(item_path):
                    print(f"  📁 {item}/")
                    # Check if this directory contains CSV files
                    csv_pattern = os.path.join(item_path, '*.csv')
                    csv_files = glob.glob(csv_pattern)
                    if csv_files:
                        print(f"     → Contains {len(csv_files)} CSV files")
                    
                    # Check subdirectories
                    try:
                        subdirs = os.listdir(item_path)
                        for subdir in subdirs:
                            if 'csv' in subdir.lower() or 'part' in subdir.lower():
                                subdir_path = os.path.join(item_path, subdir)
                                if os.path.isdir(subdir_path):
                                    csv_pattern = os.path.join(subdir_path, '*.csv')
                                    csv_files = glob.glob(csv_pattern)
                                    if csv_files:
                                        print(f"     → 📁 {subdir}/ contains {len(csv_files)} CSV files")
                                        # Show sample files
                                        for f in csv_files[:3]:
                                            print(f"        - {os.path.basename(f)}")
                                        if len(csv_files) > 3:
                                            print(f"        ... and {len(csv_files)-3} more files")
                    except PermissionError:
                        print(f"     → Cannot access subdirectories (permission denied)")
                else:
                    print(f"  📄 {item}")
        except PermissionError:
            print("  Permission denied")
    else:
        print(f"{base_dir} does not exist")
    
    # Also check if there are any filtered files in wanglab
    print("\n" + "-"*60)
    print("Checking for filtered data in wanglab:")
    wanglab_dir = '/sharefolder/wanglab'
    if os.path.exists(wanglab_dir):
        pattern = os.path.join(wanglab_dir, 'iqvia_ndc_*.csv')
        filtered_files = glob.glob(pattern)
        if filtered_files:
            print(f"Found {len(filtered_files)} filtered IQVIA files:")
            for f in sorted(filtered_files)[:5]:
                print(f"  - {os.path.basename(f)}")
            if len(filtered_files) > 5:
                print(f"  ... and {len(filtered_files)-5} more files")

if __name__ == "__main__":
    find_claims_files()