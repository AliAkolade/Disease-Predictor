# Logistic Regression

# Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import pickle

# Importing the dataset
dataset = pd.read_csv('Data/New Diabetes.csv')
X = dataset.iloc[:, :-1].values
y = dataset.iloc[:, 8:9].values

# Splitting the dataset into the Training set and Test set
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.25, random_state = 0)

# Feature Scaling
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)

# Fitting Logistic Regression to the Training set
from sklearn.linear_model import LogisticRegression
classifier = LogisticRegression(random_state = 42)
classifier.fit(X_train, y_train)

# save the model to disk
filename = 'Logistic Regression Diabetes.sav'
pickle.dump(classifier, open(filename, 'wb'))

# load the model from disk
loaded_model = pickle.load(open(filename, 'rb'))

# Predicting the Test set results
result = loaded_model.score(X_train, y_train)
print("Test score: {0:.2f} %".format(100 * result))

# Making the Confusion Matrix
from sklearn.metrics import confusion_matrix
# X_test = [[9.0,102.0,76.0,37.0,155.5482233502538,32.9,0.665,46.0]]
# X_test = sc.fit_transform(X_test)
y_pred = classifier.predict(X_train)
#print(y_pred)
cm = confusion_matrix(y_train, y_pred)
print(cm)
