# Multiple Linear Regression
# Y = c + m1*x1 + m2*x2 +...+ mn*xn
# Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Importing the dataset
dataset = pd.read_csv('50_Startups.csv')
X = dataset.iloc[:, :-1].values
y = dataset.iloc[:, 4].values

# Encoding categorical data
# Encoding the Independent Variable
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
ColumnTransformer = ColumnTransformer([('encoder', OneHotEncoder(), [3])], remainder='passthrough')
X = np.array(ColumnTransformer.fit_transform(X), dtype=np.float)

# Avoiding the Dummy Variable Trap
X = X[:, 1:]

# Splitting the dataset into the Training set and Test set
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 0)

# Feature Scaling is done by algorithm

# Fitting 2 Multiple Linear Regression to the Training set
from sklearn.linear_model import LinearRegression
regressor = LinearRegression().fit(X_train, y_train)
# regressor.fit(X_train, y_train)

# Predicting the Test set results
y_pred = regressor.predict(X_test)
print(y_pred)
print()

# Build optimal model using backwards elimination
# Its used to eliminate independent vars that aren't as important to the dependent var using a significance level
# Step 1
# Mentally set significance level to 0.05 which is 5%

import statsmodels.formula.api as sm
X = np.append(arr=np.ones((50,1)).astype(int), values=X, axis=1)  # Add column of ones(1) to the X
X_opt = X[:, [0,1,2,3,4,5]]  # List out all the indexes of the columns because you'll have to eliminate them

# Step 2 of Backward elimination
regressor_OLS = sm.OLS(endog=y, exog=X_opt).fit()

print(regressor_OLS.summary())

# Remove indexes one by one after checking the p values
# Then fit the models again and repeat
X_opt = X[:, [0, 1, 3, 4, 5]]
regressor_OLS = sm.OLS(endog=y, exog=X_opt).fit()
print(regressor_OLS.summary())

X_opt = X[:, [0, 3, 4, 5]]
regressor_OLS = sm.OLS(endog=y, exog=X_opt).fit()
print(regressor_OLS.summary())

X_opt = X[:, [0, 3, 5]]
regressor_OLS = sm.OLS(endog=y, exog=X_opt).fit()
print(regressor_OLS.summary())

X_opt = X[:, [0, 3]]
regressor_OLS = sm.OLS(endog=y, exog=X_opt).fit()
print(regressor_OLS.summary())
