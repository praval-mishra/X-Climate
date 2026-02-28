import pandas as pd 
import numpy as np
#load the CSV_-skip the header rows
df = pd.read_csv('data/hyderabad_climate.csv', skiprows=14)
# print(df.head())
# print(df.columns)
# import os 
# print(os.getcwd())

#Convert YEAR + DOY to actual data

df['DATE']=pd.to_datetime(df['YEAR'].astype(str)+df['DOY'].astype(str).str.zfill(3),format='%Y%j')

#extract Month for seasonal context
df['MONTH']=df['DATE'].dt.month

#replace -999 with Nan and handles missinng values 
df.replace(-999,np.nan,inplace=True)
df.fillna(df.mean(numeric_only=True),inplace=True)

#check the result 
print(df.head())
print(df.shape)
print(df.dtypes)

#############################################
# Feature Enginering 
# Calculate monthly statistics for each feature
# Create deviation features using Z-score by month
for col in ['T2M_MAX', 'T2M_MIN', 'T2M', 'RH2M', 'WS2M', 'PRECTOTCORR']:
    monthly_mean = df.groupby('MONTH')[col].transform('mean')
    monthly_std = df.groupby('MONTH')[col].transform('std')
    df[f'{col}_DEV'] = (df[col] - monthly_mean) / monthly_std

# Save processed data
df.to_csv('data/processed_climate_data.csv', index=False)
print("Feature engineering complete.")
print(df[['DATE', 'T2M_MAX', 'T2M_MAX_DEV', 'PRECTOTCORR', 'PRECTOTCORR_DEV']].head(10))


# Multi-class anomaly labeling
df['ANOMALY'] = 0  # Normal

# Heatwave — high max temperature deviation
df.loc[df['T2M_MAX_DEV'] > 2, 'ANOMALY'] = 1

# Cold Wave — low min temperature deviation
df.loc[df['T2M_MIN_DEV'] < -2, 'ANOMALY'] = 2

# Heavy Rainfall — high precipitation deviation
df.loc[df['PRECTOTCORR_DEV'] > 2, 'ANOMALY'] = 3

# Drought — low precipitation during normally wet months
df.loc[(df['PRECTOTCORR_DEV'] < -2) & (df['MONTH'].isin([6,7,8,9])), 'ANOMALY'] = 4

# Check distribution
print(df['ANOMALY'].value_counts())
df.to_csv('data/processed_climate_data.csv', index=False)

print(df['ANOMALY'].value_counts())
print(f"\nTotal anomaly percentage: {(df['ANOMALY'] > 0).mean()*100:.2f}%")