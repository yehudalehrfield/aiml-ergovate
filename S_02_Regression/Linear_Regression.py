# -*- coding: utf-8 -*-
"""
Linear regression analysis of a simple dataset. 

Created on Fri Dec  9 11:50:25 2022

@author: A00127096
"""

# %% Step 1: Import the required libraries and assign a shortcut for further use
import numpy as np  # numpy is a library of mathematical tools
import matplotlib.pyplot as plt  # matpotlib is a library for plotting data
import pandas as pd  # pandas ia a library for managing datasets
from sklearn.model_selection import train_test_split as tts

# call Linear Regression class from sklearn
from sklearn.linear_model import LinearRegression

# %% Step 2: Read the file and import the required data from the dataset

dataset = pd.read_csv('Cost_Data.csv')  # load csv file
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

# %% Step 7a: Fitting a linear regression model to the training set
regressor = LinearRegression()  # create a LinearRegression object
regressor.fit(x_train, y_train)  # fit the training set using linear regression

# %% Step 7b: Predict the test set results
y_pred = regressor.predict(x_test)  # predict y values from x_test dataset

# Step 8: Visualization of the results
# Training Set
plt.scatter(x_train, y_train, color='red')
y_train_pred = regressor.predict(x_train)
plt.plot(x_train, y_train_pred, color='blue')
plt.title('Distance Vs. Cost (Linear Regression) - Training')
plt.xlabel('Distance from Hub')
plt.ylabel('Cost of Maintenance')
plt.show()

# Testing Set
plt.scatter(x_test, y_test, color='red')
plt.plot(x_train, y_train_pred, color='blue')
plt.title('Distance Vs. Cost (Linear Regression) - Testing')
plt.xlabel('Distance from Hub')
plt.ylabel('Cost of Maintenance')
plt.show()
