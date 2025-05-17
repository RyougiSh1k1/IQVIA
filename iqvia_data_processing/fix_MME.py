import pandas as pd

print('Loading dataset....')
df = pd.read_csv('/sharefolder/wanglab/MME/mme_data_final_combined.csv')
print('Dataset loaded....')

df = df[df['MME'] != 0]

df = df[df['MME'] > 0]
df = df[df['dayssup'] > 0]
df['daily_MME'] = df['MME'] / df['dayssup']
# Drop rows where MME per day > 400
df = df[df['daily_MME'] <= 400]
print('Fixed MMEs....')

print("Writing to csv....")
df.to_csv('/sharefolder/wanglab/MME/mme_data_final_combined.csv', index=False)
print('Done!')
