import numpy as np
import pandas as pd
from scipy import stats

from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.feature_selection import SelectKBest
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import mean_absolute_error

from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder

test = pd.read_csv('/Users/73rabann/Desktop/DAT200/Assigments/Assignment5/test.csv')
raw_train = pd.read_csv('/Users/73rabann/Desktop/DAT200/Assigments/Assignment5/train.csv')

# drop the column 'Average Temperature During Storage (celcius)' 
train = raw_train.drop(['Average Temperature During Storage (celcius)'], axis=1)
test = test.drop(['Average Temperature During Storage (celcius)'], axis=1)

# Fill inn missing values
# Combine train and test DataFrames into a single list
data = [train, test]

# List of numeric columns
numeric_cols = ['Length (cm)', 'Width (cm)', 'Weight (g)', 'Pericarp Thickness (mm)', 'Seed Count', 'Capsaicin Content', 'Sugar Content','Moisture Content', 'Firmness']

# Fill missing values in numeric columns with the mean 
for df in data:
    for col in numeric_cols:
        df[col] = df[col].fillna(df[col].mean())

# Fill missing values in categorical columns 
categorical_cols = ['color', 'Harvest Time']
for col in categorical_cols:
    for df in data:
        df[col] = df[col].fillna(data[0][col].mode().iloc[0])

# Check for missing data
print("Missing values in train DataFrame:")
print(train.isnull().sum())

print("\nMissing values in test DataFrame:")
print(test.isnull().sum())

numeric_cols = ['Length (cm)', 'Width (cm)', 'Weight (g)', 'Pericarp Thickness (mm)', 'Seed Count', 'Capsaicin Content', 'Sugar Content', 'Firmness']
test[numeric_cols] = test[numeric_cols].fillna(test[numeric_cols].mean())

# Fill missing values in categorical columns with the mode
categorical_cols = ['color', 'Harvest Time']
test[categorical_cols] = test[categorical_cols].fillna(test[categorical_cols].mode().iloc[0])

# Define the columns to be one-hot encoded
categorical_cols = ['color', 'Harvest Time']

# Create a ColumnTransformer
ct = ColumnTransformer(
    transformers=[
        ('onehot', OneHotEncoder(drop='first', dtype=int), categorical_cols)
    ],
    remainder='passthrough',  # Remainder columns will be passed through without any transformations
    verbose_feature_names_out=False,
)

# Apply the ColumnTransformer to the 'train' DataFrame
train = pd.DataFrame(ct.fit_transform(train), columns=ct.get_feature_names_out())

categorical_cols = ['color', 'Harvest Time']

# Create a ColumnTransformer
ct = ColumnTransformer(
    transformers=[
        ('onehot', OneHotEncoder(drop='first', dtype=int), categorical_cols)
    ],
    remainder='passthrough',  # Remainder columns will be passed through without any transformations
    verbose_feature_names_out=False,
)

# Apply the ColumnTransformer to the 'test' DataFrame
test = pd.DataFrame(ct.fit_transform(test), columns=ct.get_feature_names_out())


# Using Z-scores to filter out the outliers
print(f'Shape of dataset before removing outliers: {train.shape}')
z_scores = stats.zscore(train)
abs_z_scores = np.abs(z_scores)
not_outliers = (abs_z_scores < 3).all(axis=1)
cleaned = train[not_outliers]

print(f'Shape of dataset after removing outliers: {cleaned.shape}')

# data preparation
X = cleaned.drop(columns=['Scoville Heat Units (SHU)'])
y = cleaned['Scoville Heat Units (SHU)']

# Split data into training and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.4, random_state=42)

# Define the preprocessor
preprocessor = Pipeline([
    ('scaler', StandardScaler()),
    ('feature_selection', SelectKBest())
])

# Create the pipeline
pipeline = Pipeline([
    ('preprocessor', preprocessor),
    ('regressor', LinearRegression())
])

# Define hyperparameters to tune
param_grid = {
    'preprocessor__feature_selection__k': [5,10, 15]
}

# Perform grid search with cross-validation
grid_search = GridSearchCV(pipeline, 
                           param_grid, 
                           cv=5, 
                           scoring='neg_mean_absolute_error')

grid_search.fit(X, y)

# Get the best model
best_estimator = grid_search.best_estimator_

# Make predictions
y_pred = best_estimator.predict(X_test)

# Calculate MAE
mae = mean_absolute_error(y_test, y_pred)
print("Mean Absolute Error (MAE):", mae)

# print best parameters
print(grid_search.best_params_)