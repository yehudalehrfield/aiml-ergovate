# Author: AJK
# Date: 01/08/2022
# This file will present the development of a template for the preprocessing of 
# data for further data analysis

# Step 1: Import the required libraries and assign a shortcut for further use.
import numpy as np   # numpy is a library of mathematical tools
import matplotlib.pyplot as plt   # matplotlib is a pilot for plotting charts
import pandas as pd   # pandas is a library for managing datasets

# Step 2: Read the file and import the required data from the dataset
dataset = pd.read_csv('Data.csv')   # pd.read_csv is applicable to CSV files.
X = dataset.iloc[:, :-1].values   # this command selects all rows and all columns, 
# except the last one.
y = dataset.iloc[:, -1].values   # this command selects all rows and only the 
# final column of the dataset.
# y = dataset.iloc[:, 3].values    # alternative command for loading the last column (4).

# Step 3 (optional): Perform missing data imputation to complete the dataset
from sklearn.impute import SimpleImputer   # Univariate imputer for completing 
# missing values with simple strategies.
imputer = SimpleImputer(missing_values=np.nan, strategy='mean')   # Replace missing 
# values using a descriptive statistic  (i.e. mean value)
imputer.fit(X[:, 1:3])   
X[:, 1:3] = imputer.transform(X[:, 1:3])   # The code is including the missing numbers 
# to the original  X matrix

# Step 4 (optional): Transform categorical data (into numerical values)
# Note: The process can be performed for both the depended and independent 
# values, as appropriate. 
from sklearn.compose import ColumnTransformer   # Applies transformers to columns of 
# an array or pandas DataFrame.
from sklearn.preprocessing import OneHotEncoder
ct = ColumnTransformer(transformers=[('encoder', OneHotEncoder(), [0])], remainder='passthrough')   
# here we are chosing the first column [0] to perform the transformation 
X = np.array(ct.fit_transform(X))

from sklearn.preprocessing import LabelEncoder   # Encode target labels with value 
# between 0 and n_classes-1.
le = LabelEncoder()
y = le.fit_transform(y)

# Step 5: Definition of the training and testing datesets for analysis
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 1)
# This command will split the two datasets into a training sets which will include the 80%
# of the values and a test set that will include the remaining 20% of the values.

# Step 6 (optional): Transform datase. scaling inputs
from sklearn.preprocessing import StandardScaler   # Standardize features by removing 
# the mean and scaling to unit variance.
sc = StandardScaler()   # The standard value for x is calculated as z=(x-u)
X_train[:, 3:] = sc.fit_transform(X_train[:, 3:])   # fit thee set into transform
X_test[:, 3:] = sc.transform(X_test[:, 3:])

sc_y = StandardScaler()
y_train = sc_y.fit_transform(y_train.reshape(-1,1))

# This is the end of the pre-processing template. 
# Note: Not all of the tools are needed for every analysis so understanding of 
# the dataset is crucial!