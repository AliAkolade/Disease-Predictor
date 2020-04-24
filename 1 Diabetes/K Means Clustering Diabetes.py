# K-Means Clustering

# Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import pickle

# Importing the dataset
dataset = pd.read_csv('New Diabetes.csv')
X = dataset.iloc[:, :-1].values
y = dataset.iloc[:, 8:9].values

# Splitting the dataset into the Training set and Test set
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.25, random_state = 42)


# Fitting K-Means to the dataset
from sklearn.cluster import KMeans
kmeans = KMeans(n_clusters = 2, init = 'k-means++', random_state = 42)
y_kmeans = kmeans.fit_predict(X_train)

# save the model to disk
filename = 'K Means Clustering Diabetes.sav'
pickle.dump(kmeans, open(filename, 'wb'))

# load the model from disk
loaded_model = pickle.load(open(filename, 'rb'))

# Predicting the Test set results
result = loaded_model.score(X_test, y_test)
print("Test score: {0:.2f} %".format(100 * result))
print(result)

# Making the Confusion Matrix
from sklearn.metrics import confusion_matrix
y_pred = kmeans.predict(X_test)
cm = confusion_matrix(y_test, y_pred)
print(cm)
