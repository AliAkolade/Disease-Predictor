# Importing the libraries
import numpy as np
import pandas as pd

# Importing the dataset
dataset = pd.read_csv('Diabetes.csv')

# Matrix of features
# Independent Variable Vector
X = dataset.iloc[:, :-1].values
# Dependent Variable Vector
y = dataset.iloc[:, 8:9].values


# Taking care of missing data
from sklearn.impute import SimpleImputer
imputer = SimpleImputer(missing_values=np.nan, strategy='mean')
imputer = imputer.fit(X[:, 1:8])
X[:, 1:8] = imputer.transform(X[:, 1:8])

X = pd.DataFrame(data=X)
X['Outcome'] = y
X.columns = ['Pregnancies','Glucose','BloodPressure','SkinThickness','Insulin','BMI','DiabetesPedigreeFunction','Age','Outcome']

#  Save New CSV
X.to_csv(r'New Diabetes.csv')
