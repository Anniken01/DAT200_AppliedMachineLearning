import matplotlib.pyplot as plt 
import seaborn as sns 
import pandas as pd
import numpy as np
from scipy import stats

from sklearn.model_selection import train_test_split
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import StandardScaler

raw_train = pd.read_csv('train.csv', index_col=0)
test_data = pd.read_csv('test.csv', index_col=0)

# check for missing values
print(raw_train.isnull().sum())
raw_train.shape

#raw_train.head()
print(raw_train['Diagnosis'].value_counts())

print(test_data.isnull().sum())
raw_train.shape

# convert yes and no to 1 and 0
df = raw_train
columns_to_encode = ['Alcohol_Use (yes/no)','Diabetes (yes/no)', 'Gender','Obesity (yes/no)', 'Diagnosis']
le = LabelEncoder()
for col in columns_to_encode:
    if df[col].dtype == 'object':
        df[col] = le.fit_transform(df[col])


# Using Z-scores to filter out the outliers. Z-score < |3|
print(f'Shape of dataset before removing outliers{df.shape}')
z_scores = stats.zscore(df) # calculates z score
abs_z_scores = np.abs(z_scores)
not_outliers = (abs_z_scores < 3).all(axis=1)
df = df[not_outliers]  # Update df with cleaned data
print(f'Shape of dataset after removing outliers {df.shape}')


Diagnosis = df['Diagnosis']
# Omvendt mapping dictionary
reverse_label_mapping = {
    0: 'Autoimmune Liver Diseases',
    1: 'Chirrosis',
    2: 'Drug-induced Liver Injury',
    3: 'Fatty Liver Disease',
    4: 'Healthy',
    5: 'Hepatitis',
    6: 'Liver Cancer'
}

# Konverterer de kodede diagnosene tilbake til sykdommer ved hjelp av omvendt mapping dictionary
Diagnosis = [reverse_label_mapping[label] for label in Diagnosis]

# Viser resultatene
print(Diagnosis)
