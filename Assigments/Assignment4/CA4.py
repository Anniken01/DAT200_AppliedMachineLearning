
# Importing the necessary libraries 
import pandas as pd
import numpy as np
from scipy import stats
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.decomposition import PCA
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA


# read data
raw_train = pd.read_csv('/Users/73rabann/Desktop/DAT200/Assigments/Assignment4/train.csv', index_col=0) 
raw_train.drop('index',inplace=True, axis=1) # removed one index column, since it was two
test_data = pd.read_csv('/Users/73rabann/Desktop/DAT200/Assigments/Assignment4/test.csv', index_col=0)
test_data.drop('index',inplace=True, axis=1) # removed one index, since it was two

# convert the categorical train data to numerical data
data = [raw_train, test_data]
columns_to_encode = ['Alcohol_Use (yes/no)','Diabetes (yes/no)', 'Gender','Obesity (yes/no)']
le = LabelEncoder() # create a label encoder

for df in data:
    for col in columns_to_encode:
        if df[col].dtype == 'object': # if the column is a string
            df[col] = le.fit_transform(df[col]) # convert the string to a number
df = raw_train

# Using Z-scores to filter out the outliers. Z-score < |3|
print(f'Shape of dataset before removing outliers{df.shape}')
z_scores = stats.zscore(df.drop(columns=["Diagnosis"])) # calculates z score for each column except the diagnosis column
abs_z_scores = np.abs(z_scores) # take the absolute value of the z-scores
not_outliers = (abs_z_scores < 3).all(axis=1) # if all the z-scores are less than 3, then it is not an outlier
cleaned = df[not_outliers] # remove the outliers
print(f'Shape of dataset after removing outliers {cleaned.shape}')

# Split the data into X and y, and then into training and testing sets
X = cleaned.drop(["Diagnosis"], axis=1) # contains all columns except traget column Diagnosis 
y = cleaned["Diagnosis"] # only contains the target column Diagnosis
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.4, random_state=42)

## Pipline with regularization
pipeline_reg = Pipeline([
    ('scaler', StandardScaler()),  # # Standardize the features
    ('clf', LogisticRegression(max_iter=1000))  # Classifier with regularization
])

# Define Hyperparameters
parameters_reg = {
    'clf__C' : [0.0001, 0.001, 0.01, 0.1, 1.0, 10.0, 100.0] # Regularization parameter
}

# Grid Search for Hyperparameters
grid_search_reg = GridSearchCV(estimator=pipeline_reg, # Use the pipeline as the estimator
                        param_grid=parameters_reg,     # Search over the hyperparameters defined in the dictionary
                        scoring='f1_macro',            # Use f1_macro as the scoring metric
                        cv=6,                          # Perform 6-fold cross-validation
                        n_jobs=-1)

grid_search_reg.fit(X, y) # fit the model

print(grid_search_reg.best_score_)
print(grid_search_reg.best_params_)


# Hyperparameter Tuning
best_estimator_reg = grid_search_reg.best_estimator_ # get the best estimator
best_estimator_reg.fit(X_train, y_train) # fit the best estimator
print('Accuracy: %.3f' % best_estimator_reg.score(X_test, y_test))



##  Create a pipeline with kernel
pipeline_svc = Pipeline([
    ('scaler', StandardScaler()), # Standardize the features
    ('reduce_dim', PCA()), # PCA as the dimensionality reduction technique
    ('clf', SVC())  # Classifier
])
# get parameters
pipeline_svc.get_params().keys()
# Define Hyperparameters
parameters_svc = {
    'reduce_dim__n_components': [2, 5, 10, 20, 30], # Number of components to keep
    'clf__C' : [0.0001, 0.001, 0.01, 0.1, 1.0, 10.0, 100.0], # Regularization parameter
    'clf__kernel': ['linear', 'rbf'], # Kernel type
    'clf__gamma': [0.0001, 0.001, 0.01, 0.1, 1.0, 10.0, 100.0] # Kernel coefficient
}

# Cross-Validation
grid_search_svc =GridSearchCV(estimator=pipeline_svc,  # Use the pipeline as the estimator
                        param_grid=parameters_svc,     # Search over the hyperparameters defined in the dictionary
                        scoring='f1_macro',            # Use f1_macro as the scoring metric
                        cv=6,                          # Perform 6-fold cross-validation
                        n_jobs=-1)

grid_search_svc.fit(X, y)

print(grid_search_svc.best_score_)
print(grid_search_svc.best_params_)

# Hyperparameter Tuning
best_estimator_svc = grid_search_svc.best_estimator_
best_estimator_svc.fit(X_train, y_train)
print('Accuracy: %.3f' % best_estimator_svc.score(X_test, y_test))



'''the best model was the KNN model when i did not remove the outliers.'''
# read data
raw_train = pd.read_csv('/Users/73rabann/Desktop/DAT200/Assigments/Assignment4/train.csv', index_col=0) 
raw_train.drop('index',inplace=True, axis=1) # removed one index column, since it was two
test_data = pd.read_csv('/Users/73rabann/Desktop/DAT200/Assigments/Assignment4/test.csv', index_col=0)
test_data.drop('index',inplace=True, axis=1) # removed one index, since it was two

# convert the categorical train data to numerical data
data = [raw_train, test_data]
columns_to_encode = ['Alcohol_Use (yes/no)','Diabetes (yes/no)', 'Gender','Obesity (yes/no)']
le = LabelEncoder() # create a label encoder

for df in data:
    for col in columns_to_encode:
        if df[col].dtype == 'object': # if the column is a string
            df[col] = le.fit_transform(df[col]) # convert the string to a number

X = raw_train.drop(["Diagnosis"], axis=1) # contains all columns except traget column Diagnosis 
y = raw_train["Diagnosis"] # only contains the traget column Diagnosis
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.4, random_state=42)


pipe_knn = Pipeline([('scaler', StandardScaler()), 
                     ('lda', LDA()), 
                     ('knn', KNeighborsClassifier())])

# Define a dictionary of hyperparameters to search over
param_grid_knn = {'knn__n_neighbors': [5, 10, 15, 20, 25],
                  'knn__leaf_size': [10, 20, 30, 40, 50],
                  'knn__weights': ['uniform', 'distance'],
                  'knn__p': [1, 2], # L1/L2
                  }

# Use GridSearchCV to search over hyperparameters for the KNeighborsClassifier
grid_search_knn = GridSearchCV(estimator=pipe_knn,        # Use the pipeline as the estimator
                        param_grid=param_grid_knn, # Search over the hyperparameters defined in the dictionary
                        scoring='f1_macro',        # Use f1_macro as the scoring metric
                        cv=5,                      # Perform 6-fold cross-validation
                        n_jobs=-1)                 

# Fit the grid search object on the training data
grid_search_knn.fit(X, y)

# Print the best score and best set of hyperparameters
print('%.3f'% grid_search_knn.best_score_)
print('the best parameters:',grid_search_knn.best_params_)

# Hyperparameter Tuning
best_estimator_knn = grid_search_knn.best_estimator_
best_estimator_knn.fit(X_train, y_train)
print('Accuracy: %.3f' % best_estimator_knn.score(X_test, y_test))