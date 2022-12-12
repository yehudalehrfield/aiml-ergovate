# -*- coding: utf-8 -*-
"""
SVM Regression analysis on a simple dataset

Created on Fri Dec  9 12:29:49 2022

@author: A00127096
"""

# %% Step 1: Import the required libraries and assign a shortcut for further use
import numpy as np  # numpy is a library of mathematical tools
import matplotlib.pyplot as plt  # matpotlib is a library for plotting data
import pandas as pd  # pandas ia a library for managing datasets
from sklearn.model_selection import train_test_split as tts
# standardize features by removing the mean and scaling to unit variance
from sklearn.preprocessing import StandardScaler
# call SVR class from sklearn
from sklearn.svm import SVR

# %% Step 2: Read the file and import the required data from the dataset
dataset = pd.read_csv('Cost_Data.csv')  # load csv file
x = dataset.iloc[:, :-1].values  # select all rows of all but last columns
y = dataset.iloc[:, -1].values  # select all rows of only the last column

# %% Step 3 (optional): Perform missing data imputation to complete the dataset
# Not relevant to this example

# %% Step 4 (optional): Transform categorical data into numerical values.
# Not relevant to this example

# %% Step 5: Definition of the training and testing datasets for analysis
# Small dataset, so we won't split into subsets

# %% Step 6 (optional): Transform dataset and scale the inputs.
sc = StandardScaler()  # standard value for x is calculated as z=(x-u)
x = sc.fit_transform(x)  # fit the set and tranform
y = sc.fit_transform(y.reshape(-1, 1))

# %% Step 7: Fitting an SVR REgression Model to the dataset
sv_regressor = SVR(kernel='rbf')  # create an SVR object
sv_regressor.fit(x, y)  # fit the SVM regressor to the dataset

# %% Step 8: Predict a vlue for a given y
# need to transfrom as above (dataset)
y_pred_SVR = sv_regressor.predict(sc.transform([[40]]))
y_pred_SVR = sc.inverse_transform(y_pred_SVR.reshape(-1, 1))

# Step 9: Visualization of the results
# SVR Regression
plt.scatter(x, y, color='red')
y_pred = sv_regressor.predict(x)
plt.plot(x, y_pred, color='blue')
plt.title('Distance Vs. Cost (SVR Regression)')
plt.xlabel('Distance from Hub')
plt.ylabel('Cost of Maintenance')
plt.show()
