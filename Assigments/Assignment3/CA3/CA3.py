import pandas as pd 
import numpy as np 
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from scipy import stats
from sklearn.preprocessing import StandardScaler

raw_test = pd.read_csv('test.csv')
raw_train = pd.read_csv('train.csv')
train = raw_train.dropna() # remove NaN values
train = train.iloc[:,1:] # create new train df without the first column
test = raw_test.iloc[:,1:]# create new test df without the first colum

#remove outliers with z-score, z < |3|
z_scores = stats.zscore(train) # calculates the z score
abs_z_scores = np.abs(z_scores)
not_outliers = (abs_z_scores < 3).all(axis=1)
cleaned = train[not_outliers]

X = cleaned.drop(["Edible"], axis=1) # contains all columns except traget colum Edible 
y = cleaned["Edible"] # only contains the traget colum Edible
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.4, random_state=42)

# scale data
sc = StandardScaler()
sc.fit(X_train)
X_train_sc = sc.transform(X_train)
X_test_sc = sc.transform(X_test)

# Parameters to opimize the classifier (used scikit-learn.org to find potensially parameters)
criterions = ['entropy', 'gini', 'log_loss'] # functions that measures the quality of the split
values_n_estimator = [100, 250, 500, 750, 1000] # number of trees in the forest

best_accuracy = 0
best_parameters = {} # stores the best parameters

for criterion in criterions:
    for n_estimator in values_n_estimator:
        forest = RandomForestClassifier(criterion=criterion, n_estimators=n_estimator,random_state=42,n_jobs=-1,bootstrap=True)
        
        forest.fit(X_train_sc, y_train) # Fit the classifier on the training data
        y_pred = forest.predict(X_test_sc) # Make predictions on the test data
        accuracy = accuracy_score(y_test, y_pred) # Calculate accuracy
        
        # check if the current model is the best, and update with which parameters that is the best
        if accuracy > best_accuracy:
            best_accuracy = accuracy
            best_parameters = {'criterion': criterion, 'n_estimators': n_estimator}

print("Best Parameters:", best_parameters)
print(f'Accuracy: {accuracy:.3f}')


# Using the best parameters from training
forest = RandomForestClassifier(criterion='gini',
                                n_estimators=750,
                                random_state=42,
                                n_jobs=-1,
                               bootstrap=True
                               )
forest.fit(X, y)

submission = forest.predict(test)