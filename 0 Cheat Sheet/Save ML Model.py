"""Save Model Using Pickle (File)"""
import pickle

model = LogisticRegression()
model.fit(X_train, Y_train)

# save the model to disk
filename = 'finalized_model.sav'
pickle.dump(model, open(filename, 'wb'))

# load the model from disk
loaded_model = pickle.load(open(filename, 'rb'))
result = loaded_model.score(X_test, Y_test)
print(result)


""" Save Model Using joblib """
import joblib

# Fit the model on training set
model = LogisticRegression()
model.fit(X_train, Y_train)

# save the model to disk
filename = 'finalized_model.sav'
joblib.dump(model, filename)

# load the model from disk
loaded_model = joblib.load(filename)
result = loaded_model.score(X_test, Y_test)
print(result)

"""Save Model Using Pickle (String)"""
import pickle

knn.fit(X_train, y_train)
# Save the trained model as a pickle string.
saved_model = pickle.dumps(knn)
print(saved_model)

# Load the pickled model
knn_from_pickle = pickle.loads(saved_model)

# Use the loaded pickled model to make predictions
knn_from_pickle.predict(X_test)

score = Pickled_LR_Model.score(Xtest, Ytest)
# Print the Score
print("Test score: {0:.2f} %".format(100 * score))