# Decision Tree Classification

# Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import pickle


# Importing the dataset
dataset = pd.read_csv('Data/New Diabetes.csv')
X = dataset.iloc[:, :-1].values
y = dataset.iloc[:, 8:9].values
#
# Splitting the dataset into the Training set and Test set
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.25, random_state = 0)

# Feature Scaling
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)

# # Fitting Decision Tree Classification to the Training set
# from sklearn.tree import DecisionTreeClassifier
# classifier = DecisionTreeClassifier(criterion = 'entropy', random_state = 0)
# classifier.fit(X_train, y_train)
#
# # save the model to disk
# filename = 'Models/Decision Tree Diabetes.sav'
# pickle.dump(classifier, open(filename, 'wb'))

# load the model from disk
loaded_model = pickle.load(open('Models/Decision Tree Diabetes.sav', 'rb'))

# Predicting the Test set results
result = loaded_model.score(X_test, y_test)
print("Test score: {0:.2f} %".format(100 * result))

# Making the Confusion Matrix
from sklearn.metrics import confusion_matrix
y_pred = loaded_model.predict(X_test)
cm = confusion_matrix(y_test, y_pred)
print(cm)
