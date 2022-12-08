# -*- coding: utf-8 -*-
"""
This file will present the development of a template for the preprocessing of
data for further analysis

Created on Wed Dec  7 14:40:54 2022

@author: A00127096
"""

# %% Step 1: Import the required libraries and assign a shortcut for further use
import numpy as np  # numpy is a library of mathematical tools
import matplotlib.pyplot as plt  # matpotlib is a library for plotting data
import pandas as pd  # pandas ia a library for managing datasets
# Univariate imputer for comnpleting missing values with simple strategies
from sklearn.impute import SimpleImputer
# Applies transformation to columns with categorical data
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
# encode target labels with values between 0 and n_classes-1
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split as tts
# standardize features by removing the mean and scaling to unit variance
from sklearn.preprocessing import StandardScaler

# %% Step 2: Read the file and import the required data from the dataset

dataset = pd.read_csv('Data.csv')  # load csv file
x = dataset.iloc[:, :-1].values  # select all rows of all but last columns
y = dataset.iloc[:, -1].values  # select all rows of only the last column
x_original = x.copy()  # keeping track of changes
y_original = y.copy()  # keeping track of changes

# %% Step 3 (optional): Perform missing data imputation to complete the dataset

# replace missing values using a descriptive strategy
imputer = SimpleImputer(missing_values=np.nan, strategy='mean')
imputer.fit(x[:, 1:3])
# put missing values into the original x matrix
x[:, 1:3] = imputer.transform(x[:, 1:3])
x_fill_in_missing = x.copy()  # keeping track of changes

# %% Step 4 (optional): Transform categorical data into numerical values.

ct = ColumnTransformer(transformers=[('encoder', OneHotEncoder(), [
                       0])], remainder='passthrough')  # choose first column
x = np.array(ct.fit_transform(x))  # update x with transformed categories
x_categorical_to_numeric = x.copy()  # keeping track of changes

le = LabelEncoder()
y = le.fit_transform(y)
y_categorical_to_numeric = y.copy()  # keeping track of changes

# %% Step 5: Definition of the training and testing datasets for analysis

# split the two datasets into a training set using the 80-20 rule
# 80% training and 20% testing
x_train, x_test, y_train, y_test = tts(x, y, test_size=0.2, random_state=1)
x_train_orig, x_test_orig, y_train_orig, y_test_orig = x_train.copy(
), x_test.copy(), y_train.copy(), y_test.copy()

# %% Step 6 (optional): Transform dataset and scale the inputs.
ss = StandardScaler()  # standard value for x is calculated as z=(x-u)
x_train[:, 3:] = ss.fit_transform(x_train[:, 3:])  # fit the set into tranform
x_test[:, 3:] = ss.fit_transform(x_test[:, 3:])
y_train = ss.fit_transform(y_train.reshape(-1, 1))
