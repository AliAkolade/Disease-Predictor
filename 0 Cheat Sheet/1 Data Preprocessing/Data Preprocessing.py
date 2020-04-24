# 1 Data Preprocessing Template

# Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Importing the dataset
dataset = pd.read_csv('Data.csv')
# Matrix of features
X = dataset.iloc[:, :-1].values  # [Row,Column] ... [_:_ , _:_]
# Dependent Variable Vector
y = dataset.iloc[:, 3].values


# Taking care of missing data
from sklearn.impute import SimpleImputer
imputer = SimpleImputer(missing_values=np.nan, strategy='mean')
imputer = imputer.fit(X[:, 1:3]) #1:3 takes 1 - 2..  Upperbound is always - 1
X[:, 1:3] = imputer.transform(X[:, 1:3])
"""Others"""

#Ignore the data row
# Will drop all rows that have any missing values.
dataframe.dropna(inplace=True)
#You can also select to drop the rows only if all of the values in the row are missing.
dataframe.dropna(how='all',inplace=True)
#Sometimes, you may just want to drop a column (variable) that has some missing values.
dataframe.dropna(axis=1,inplace=True)
#Finally, you may want to keep only the rows with at least 4 non-na values:
dataframe.dropna(thresh=4,inplace=True)

#Back-fill or forward-fill to propagate next or previous values respectively:
#for back fill
dataframe.fillna(method='bfill',inplace=True)
#for forward-fill
dataframe.fillna(method='ffill',inplace=True)
#Note that the NaN value will remain even after forward filling or back filling if a next or previous value isn’t available or it is also a NaN value.

#Replace with some constant value outside fixed value range-999,-1 etc
dataframe.Column_Name.fillna(-99,inplace=True)

#MEAN: Suitable for continuous data without outliers
dataframe.Column_Name.fillna(dataframe.Column_Name.mean(),inplace=True)
#MEDIAN :Suitable for continuous data with outliers
dataframe.Column_Name.fillna(dataframe.Column_Name.median(),inplace=True)



# Encoding categorical data
# Encoding the Independent Variable
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
ColumnTransformer = ColumnTransformer([('encoder', OneHotEncoder(), [0])], remainder='passthrough')
X = np.array(ColumnTransformer.fit_transform(X), dtype=np.float)

# Encoding the Dependent Variable
labelencoder_y = LabelEncoder()
y = labelencoder_y.fit_transform(y)

# Splitting the dataset into the Training set and Test set
from sklearn.model_selection import train_test_split

# 0.2 = 20% , Random state is normally 42
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)


# Feature Scaling
# Done so that a feature wont overshadow another due to its size
from sklearn.preprocessing import StandardScaler

sc_X = StandardScaler()
X_train = sc_X.fit_transform(X_train)
X_test = sc_X.transform(X_test)
# Feature Scaling can also be done for the y axis but its not needed here
print()
