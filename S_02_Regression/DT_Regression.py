# -*- coding: utf-8 -*-
"""
Decision Tree Regression analysis of a simple dataset

Created on Mon Dec 12 10:30:58 2022

@author: A00127096
"""

# %% Step 1: Import the required libraries and assign a shortcut for further use
import numpy as np  # numpy is a library of mathematical tools
import matplotlib.pyplot as plt  # matpotlib is a library for plotting data
import pandas as pd  # pandas ia a library for managing datasets
from sklearn.model_selection import train_test_split as tts
# standardize features by removing the mean and scaling to unit variance
from sklearn.preprocessing import StandardScaler
# call Decision Tree Regression class from sklearn
from sklearn.tree import DecisionTreeRegressor

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

# %% Step 7: Fitting the Decision Tree Regression model to the dataset
regressor = DecisionTreeRegressor(random_state=0)
regressor.fit(x, y)  # fit the regressor model to the dataset

# %% Setp 8: Predict a value for a given y
y_pred = regressor.predict([[40]])

# %% Step 9: Visualizing thr results
# Decision Tree Regression
plt.figure(1)
# Discretize the x domain (increase resolution of results)
x_grid = np.arange(min(x), max(x), 0.01)
x_grid = x_grid.reshape((len(x_grid), 1))  # Reshape the vector

plt.scatter(x, y, color='red')
y_pred = regressor.predict(x_grid)
plt.plot(x_grid, y_pred, color='blue')
plt.title('Distance Vs. Cost (Decision Tree Regression)')
plt.xlabel('Distance from Hub')
plt.ylabel('Cost of Maintenance')
plt.show()
