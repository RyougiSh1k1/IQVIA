### This file filters IQVIA based on ndc code and stores it to a certain directory

import pandas
import time
import os
from iqvia_data_processing.read_iqvia import read_iqvia_header, read_iqvia_claims, read_ndc_codes

header_data = read_iqvia_header()
ndc_codes = read_ndc_codes()

years = [str(year) for year in range(2006, 2023)]

for year in years:
    start_time = time.time()
    iqvia_data = read_iqvia_claims(year, header_data, ndc_codes)
    iqvia_data.to_csv(f'/sharefolder/wanglab/iqvia_ndc_{year}.csv', index = False)
    end_time = time.time()
    elapsed_time = end_time - start_time
    print(f"Elapsed Time: {elapsed_time:.2f} seconds", flush = True)
    print('Done with year: ',year, "!",flush = True)
