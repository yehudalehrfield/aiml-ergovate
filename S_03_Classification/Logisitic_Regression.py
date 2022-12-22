# -*- coding: utf-8 -*-
"""
Logistic Regression Classification of a simple dataset

Created on Thu Dec 15 09:09:38 2022

@author: A00127096
"""

# %% Step 1: Import the required libraries and assign a shortcut for further use
from sklearn.metrics import confusion_matrix, accuracy_score
import numpy as np  # numpy is a library of mathematical tools
import matplotlib.pyplot as plt  # matpotlib is a library for plotting data
import pandas as pd  # pandas ia a library for managing datasets
from sklearn.model_selection import train_test_split as tts
# standardize features by removing the mean and scaling to unit variance
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from matplotlib.colors import ListedColormap

# %% Step 2: Read the file and import the required data from the dataset

dataset = pd.read_csv('SLExtension.csv')  # load csv file
x = dataset.iloc[:, :-1].values  # select all rows of all but last columns
y = dataset.iloc[:, -1].values  # select all rows of only the last column

# %% Step 3 (optional): Perform missing data imputation to complete the dataset
# This step is not applicable to this example

# %% Step 4 (optional): Transform categorical data into numerical values.
# This step is not applicable to this example

# %% Step 5: Definition of the training and testing datasets for analysis

# split the two datasets into a training set using the 80-20 rule
# 80% training and 20% testing
x_train, x_test, y_train, y_test = tts(x, y, test_size=0.25, random_state=0)

# %% Step 6 (optional): Transform dataset and scale the inputs.
sc = StandardScaler()  # standard value for x is calculated as z=(x-u)
x_train = sc.fit_transform(x_train)  # fit the set into tranform
x_test = sc.transform(x_test)

# %% Step 7a: Train the Logistic Regression model on the training set

classifier = LogisticRegression(random_state=0)  # create Logistic Reg object
classifier.fit(x_train, y_train)  # fit regression model to the training set

# %% Step 7b: Predict the test set results
y_pred_value = classifier.predict(sc.transform(
    [[601826, 29]]))  # predict for a given y
y_pred = classifier.predict(x_test)  # predict for the testing set

# %% Step 8: Calcualte the Confusion Matrix
cm = confusion_matrix(y_test, y_pred)
acc_score = accuracy_score(y_test, y_pred)

# %% Step 9: Visualization

# Training set
x_set, y_set = x_train, y_train
x1, x2 = np.meshgrid(
    np.arange(
        start=x_set[:, 0].min() - 1,
        stop=x_set[:, 0].max() + 1,
        step=0.01
    ),
    np.arange(
        start=x_set[:, 1].min() - 1,
        stop=x_set[:, 1].max() + 1,
        step=0.01)
)
plt.contourf(
    x1,
    x2,
    classifier
    .predict(np.array([x1.ravel(), x2.ravel()]).T)
    .reshape(x1.shape), alpha=0.5, cmap=ListedColormap(('red', 'green'))
)
plt.xlim(x1.min(), x1.max())
plt.ylim(x2.min(), x2.max())
for i, j in enumerate(np.unique(y_set)):
    plt.scatter(x_set[y_set == j, 0], x_set[y_set == j, 1],
                c=ListedColormap(('red', 'green'))(i), label=j)
plt.title('Logistic Regression Classification (Training set)')
plt.xlabel('CAPEX investment')
plt.ylabel('Number of O&M employees')
plt.legend()
plt.show()

# Test set
x_set, y_set = x_test, y_test
x1, x2 = np.meshgrid(
    np.arange(
        start=x_set[:, 0].min() - 1,
        stop=x_set[:, 0].max() + 1,
        step=0.01
    ),
    np.arange(
        start=x_set[:, 1].min() - 1,
        stop=x_set[:, 1].max() + 1,
        step=0.01)
)
# Fill in regions
plt.contourf(
    x1,
    x2,
    classifier
    .predict(np.array([x1.ravel(), x2.ravel()]).T)
    .reshape(x1.shape), alpha=0.5, cmap=ListedColormap(('red', 'green')))
plt.xlim(x1.min(), x1.max())
plt.ylim(x2.min(), x2.max())
# Plot classified points
for i, j in enumerate(np.unique(y_set)):
    plt.scatter(x_set[y_set == j, 0], x_set[y_set == j, 1],
                c=ListedColormap(('red', 'green'))(i), label=j)
plt.title('Logistic Regression Classification (Test set)')
plt.xlabel('CAPEX investment')
plt.ylabel('Number of O&M employees')
plt.legend()
plt.show()
