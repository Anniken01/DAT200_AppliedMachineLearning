import pandas as pd 

# Load the iris dataset
iris = 'https://archive.ics.uci.edu/ml/machine-learning-databases/iris/iris.data'


iris_pd = pd.read_csv(iris, header=None)

# Set the column names
column_names = ['sepal_length', 'sepal_width', 'petal_length',
'petal_width', 'types']


iris.columns = column_names

# set the row names
row_names = ['flower_{}'.format(i) for i in range(1, 151)]

