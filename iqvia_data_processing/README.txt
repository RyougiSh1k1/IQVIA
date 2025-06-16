# IQVIA Data Processing Documentation: AIM-AHEAD PROJECT

This documentation provides an overview of the data processing steps for the IQVIA dataset, aimed at extracting features for the AIM-AHEAD project's ML model.

## Objective

The goal is to extract the following features from the dataset:

1. **Total Morphine Milligram Equivalent (MME)** prescribed during the most recent 365 days, including MMEs from MOUDs when dispensed to the patient.
2. **Number of pharmacies** where narcotics and/or sedatives were filled in the last 2 years.   ### NOT INCLUDE / NOT FOUND
3. **Total MMEs** prescribed in the last 2 years.
4. **Opioid prescriptions** with daily MMEs >120 in the last 2 years.
5. **Number of prescribers** where opioids were obtained in the last 2 years.
6. **Total MMEs prescribed for 1+ years** before the current date, including MMEs from MOUDs.
7. **Number of prescribers** where opioids were obtained in the last 180 days.

Features are categorized into three main groups: **MME**, **prescriber**, and **pharmacy** (NO PHARMACY DATA FOUND). These are stored in separate CSV datasets with primary keys `pat_id` (patient ID) and `most_recent_date` (the latest pickup date).

> **Note**: Currently, no pharmacy-related features are included due to missing pharmacy information.
We are expecting around 3,851,892 observations.

Things to do Next:

1. Merge both MME data and Prescriber data to be final input feature data.
2. Extract ICD labels for classifying patients with having an OUD event or not.
3. Train ML model on finished feature set.

---

## Files Summary

- **`0_filter_data.py`**: Filters raw IQVIA data for opioid-specific NDC codes.
  - **Input**: Raw IQVIA data (2006-2022), `ndc_codes.txt`
  - **Output**: `iqvia_ndc_{year}.csv` (filtered data by year)

- **`1_mme_conversion.py`**: Matches IQVIA prescriptions to extract and calculate MMEs.
  - **Input**: `iqvia_ndc_{year}.csv`
  - **Output**: `mme_data_final_{year}.csv` (MME data by year), `ndc_not_found.txt` (NDC codes missing in `OPIOID_FINDER.csv`)

- **`2_combine_MME_parts.py`**: Combines all yearly MME datasets into one.
  - **Input**: `mme_data_final_{year}.csv`
  - **Output**: `mme_data_final_combined.csv`

- **`3_extract_MME_features.py`**: Extracts MME-related features for the ML model.
  - **Input**: `mme_data_final_combined.csv`
  - **Output**: `MME_features_200_max.csv` (filtered by a daily MME max of 200)

- **`4_extract_prescriber.py`**: Extracts prescriber-related features.
  - **Input**: `mme_data_final_combined.csv`
  - **Output**: `prscbr_features.csv`

- **`5_merge_features.py`**: Merge MME and Prescriber features.
  - **Input**: 
  - **Output**: 

- **`6_extract_OUD_labels.py`**: Assign OUD labels (0/1) based on the matching results of ICD codes.
  - **Input**: 
  - **Output**: 

- **`7_merge_OUD.py`**: Merge OUD labels with input predictive features.
  - **Input**: 
  - **Output**: 

- **`8_ML_training.py`**: Train 3 ML models (LR, RF, XGBoost) usign cross-entropy loss.
  - **Input**: 
  - **Output**:

- **`9_NN_training.py`**: Train 3 NN models usign cross-entropy loss.
  - **Input**: 
  - **Output**: 

- **`fix_MME.py`**: Applies any necessary adjustments to the final MME dataset.
  - **Input**: `mme_data_final_combined.csv`
  - **Output**: *overwrites `mme_data_final_combined.csv`*

- **`ndc_codes.txt`**: NDC codes used for filtering opioid data.

- **`OPIOID_FINDER.csv`**: Contains medication details by NDC code and relevant MME info.

- **`read_iqvia.py`**: Helper function for reading raw data files.

---

Ryan Stofer  
2024
