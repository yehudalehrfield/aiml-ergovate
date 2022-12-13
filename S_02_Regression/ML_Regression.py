# -*- coding: utf-8 -*-
"""
Multiple Linear Regression analysis

Created on Mon Dec 12 14:52:39 2022

@author: A00127096
"""

# %% Step 1: Import the required libraries and assign a shortcut for further use
import numpy as np  # numpy is a library of mathematical tools
import matplotlib.pyplot as plt  # matpotlib is a library for plotting data
import pandas as pd  # pandas ia a library for managing datasets
from sklearn.model_selection import train_test_split as tts
# standardize features by removing the mean and scaling to unit variance
from sklearn.linear_model import LinearRegression

# %% Step 2: Read the file and import the required data from the dataset

dataset = pd.read_csv('kc_house_data.csv')  # load csv file
x = dataset.iloc[:, :-1].values  # select all rows of all but last columns
y = dataset.iloc[:, -1].values  # select all rows of only the last column

# %% Step 3 (optional): Perform missing data imputation to complete the dataset
# Not relevant to this example

# %% Step 4 (optional): Transform categorical data into numerical values.
# Not relevant to this example

# %% Step 5: Definition of the training and testing datasets for analysis

# split the two datasets into a training set using the 80-20 rule
# 80% training and 20% testing
x_train, x_test, y_train, y_test = tts(x, y, test_size=0.2, random_state=1)

# %% Step 6 (optional): Transform dataset and scale the inputs.
# Not relevant to this example

# %% Step 7: Fitting a Simple Linear Regression Model to the training set
regressor = LinearRegression()
regressor.fit(x_train, y_train)

# %% Step 7b: Predict the test set results
y_pred = regressor.predict(x_test)  # predict values for y values given x

# Step 8: Visualization of the results
# Training Set
x_train_range = np.arange(0, len(y_train))
plt.scatter(x_train_range, y_train, color='red')
y_pred_train = regressor.predict(x_train)
plt.scatter(x_train_range, y_pred_train, marker='1', alpha=.5)
plt.xlabel('House #')
plt.ylabel('Price')
plt.title('Training Set')
plt.show()

# Testing Set
x_test_range = np.arange(0, len(y_test))
plt.scatter(x_test_range, y_test, color='red')
y_pred_test = regressor.predict(x_test)
plt.scatter(x_test_range, y_pred_test, marker='1', alpha=.5)
plt.xlabel('House #')
plt.ylabel('Price')
plt.title('Testing Set')
plt.show()
