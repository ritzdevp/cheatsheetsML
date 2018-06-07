# Data Preprocessing Template

# Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Importing the dataset
dataset = pd.read_csv('Data.csv')   
X = dataset.iloc[:, :-1].values 
"""All rows. All columns except last one,as its the outcome"""

y = dataset.iloc[:, 3].values
"""All rows and last column"""


#Taking care of missing values
from sklearn.preprocessing import Imputer
imputer = Imputer(missing_values='NaN', strategy = 'mean', axis = 0)
imputer = imputer.fit(X[:, 1:3])
X[:, 1:3] = imputer.transform(X[:, 1:3])

#Handling categorical variables
#using label encoder
"""from sklearn.preprocessing import LabelEncoder
labelencoder_X = LabelEncoder()
X[:,0] = labelencoder_X.fit_transform(X[:,0]) #countries"""
#There is a prob tho. This label encoder will give values
#like Germany = 1 Spain = 2. Which is not allowed as Spain!>Germany lol

# Dummy encoding is better.
#ONE HOT ENCODER
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
labelencoder_X = LabelEncoder()
X[:,0] = labelencoder_X.fit_transform(X[:,0]) #countries
onehotencoder = OneHotEncoder(categorical_features = [0]) #0th col is to be encoded
X = onehotencoder.fit_transform(X).toarray();

#labelencoder as y contains only two values yes and no ie 1 and 0
labelencoder_y = LabelEncoder()
y = labelencoder_y.fit_transform(y)



# Splitting the dataset into the Training set and Test set
from sklearn.cross_validation import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 0)
#random state = 42 :P The nice number :P norm



# Feature Scaling
"""from sklearn.preprocessing import StandardScaler
sc_X = StandardScaler()
X_train = sc_X.fit_transform(X_train)
X_test = sc_X.transform(X_test)
sc_y = StandardScaler()
y_train = sc_y.fit_transform(y_train)"""
#For test set we don;t need to fit sc_X object because it is already fit to training set
#we might need to scale y too. Depending upon the situation.
