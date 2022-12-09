# -*- coding: utf-8 -*-

"""
Polynomial Regression analysis of a simple dataset. 

Created on Wed Dec  7 14:40:54 2022

@author: A00127096
"""

# %% Step 1: Import the required libraries and assign a shortcut for further use
import numpy as np  # numpy is a library of mathematical tools
import matplotlib.pyplot as plt  # matpotlib is a library for plotting data
import pandas as pd  # pandas ia a library for managing datasets
from sklearn.model_selection import train_test_split as tts
# standardize features by removing the mean and scaling to unit variance
from sklearn.preprocessing import StandardScaler
# call Linear Regression class from sklearn
from sklearn.linear_model import LinearRegression

# call Linear and Polynomial Regression classes from sklearn
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures

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
# Not relevant to this example

# %% Step 7a: Fitting a linear regression model to the dataset
lin_reg = LinearRegression()  # create a LinearRegression object
lin_reg.fit(x, y)  # fit the dataset using linear regression

# %% Step 7b: Fitting a polynomial regression model to the dataset
# We treat the polynomial regression as linear by transforming
poly_reg = PolynomialFeatures(degree=4)  # define the polynomial order
x_poly = poly_reg.fit_transform(x)  # fit and transform data
lin_reg_2 = LinearRegression()  # create a Linear Regression object
lin_reg_2.fit(x_poly, y)  # fit the regression model to the dataset

# %% Step 8: Predict a value for a given x
# Linear Regression
y_pred_lin_40 = lin_reg.predict([[40]])  # [[]] denotes a single number
y_pred_poly_40 = lin_reg_2.predict(poly_reg.fit_transform([[40]]))

# Step 9: Visualization of the results
# Linear Regression
y_pred_lin = lin_reg.predict(x)
plt.scatter(x, y, color='red')
plt.plot(x, y_pred_lin, color='blue')
plt.title('Distance Vs. Cost (Linear Regression)')
plt.xlabel('Distance from Hub')
plt.ylabel('Cost of Maintenance')
plt.show()

# Polynomial Regression
y_pred_poly = lin_reg_2.predict(x_poly)
plt.scatter(x, y, color='red')
plt.plot(x, y_pred_poly, color='blue')
plt.title('Distance Vs. Cost (Polynomial Regression)')
plt.xlabel('Distance from Hub')
plt.ylabel('Cost of Maintenance')
plt.show()
