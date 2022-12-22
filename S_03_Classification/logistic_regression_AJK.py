# Author: AJK
# Date: 01/08/2022
# This file will present the development of a template for Logistic Regression Classification

# Step 1: Import the required libraries and assign a shortcut for further use.
import numpy as np   # numpy is a library of mathematical tools
import matplotlib.pyplot as plt   # matplotlib is a pilot for plotting charts
import pandas as pd   # pandas is a library for managing datasets

# Step 2: Read the file and import the required data from the dataset
dataset = pd.read_csv('SLExtension.csv')   # pd.read_csv is applicable to CSV files.
X = dataset.iloc[:, :-1].values   # this command selects all rows and all columns, 
# except the last one.
y = dataset.iloc[:, -1].values   # this command selects all rows and only the 
# final column of the dataset.

# Step 3 (optional): Perform missing data imputation to complete the dataset
# Note: This step is not required for this dataset.

# Step 4 (optional): Transform categorical data (into numerical values)
# Note: This step is not required for this dataset.

# Step 5: Definition of the training and testing datesets for analysis
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.25, random_state = 0)
# This command will split the two datasets into a training sets which will include the 80%
# of the values and a test set that will include the remaining 20% of the values.

# Step 6 (optional): Transform database and scale inputs
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)

# Step 7: Train the Logistic Regression model on Training set
from sklearn.linear_model import LogisticRegression # From the sklearn library 
# we call the LinearRegression class
classifier = LogisticRegression(random_state = 0) # We create an object of the LinearRegression 
# class which will return an object
classifier.fit(X_train, y_train) # We fit the regression model to the training set

# Step 7b: Predict the test set results
y_pred_value = classifier.predict(sc.transform([[601826,	29]]))

y_pred = classifier.predict(X_test)

# Step 7b: Calculate the Confusion Matrix
from sklearn.metrics import confusion_matrix, accuracy_score
cm = confusion_matrix(y_test, y_pred)
accscore=accuracy_score(y_test, y_pred)

# Step 8 Visualisation of the results
# Training set
from matplotlib.colors import ListedColormap
X_set, y_set = X_train, y_train
X1, X2 = np.meshgrid(np.arange(start = X_set[:, 0].min() - 1, stop = X_set[:, 0].max() + 1, step = 0.01),
                     np.arange(start = X_set[:, 1].min() - 1, stop = X_set[:, 1].max() + 1, step = 0.01))
plt.contourf(X1, X2, classifier.predict(np.array([X1.ravel(), X2.ravel()]).T).reshape(X1.shape),
             alpha = 0.75, cmap = ListedColormap(('red', 'green')))
plt.xlim(X1.min(), X1.max())
plt.ylim(X2.min(), X2.max())
for i, j in enumerate(np.unique(y_set)):
    plt.scatter(X_set[y_set == j, 0], X_set[y_set == j, 1],
                c = ListedColormap(('red', 'green'))(i), label = j)
plt.title('Logistic Regression (Training set)')
plt.xlabel('CAPEX investment')
plt.ylabel('Number of O&M employees')
plt.legend()
plt.show()

# Test set
from matplotlib.colors import ListedColormap
X_set, y_set = X_test, y_test
X1, X2 = np.meshgrid(np.arange(start = X_set[:, 0].min() - 1, stop = X_set[:, 0].max() + 1, step = 0.01),
                     np.arange(start = X_set[:, 1].min() - 1, stop = X_set[:, 1].max() + 1, step = 0.01))
plt.contourf(X1, X2, classifier.predict(np.array([X1.ravel(), X2.ravel()]).T).reshape(X1.shape),
             alpha = 0.75, cmap = ListedColormap(('red', 'green')))
plt.xlim(X1.min(), X1.max())
plt.ylim(X2.min(), X2.max())
for i, j in enumerate(np.unique(y_set)):
    plt.scatter(X_set[y_set == j, 0], X_set[y_set == j, 1],
                c = ListedColormap(('red', 'green'))(i), label = j)
plt.title('Logistic Regression (Test set)')
plt.xlabel('CAPEX investment')
plt.ylabel('Number of O&M employees')
plt.legend()
plt.show()