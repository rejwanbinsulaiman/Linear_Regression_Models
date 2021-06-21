# -*- coding: utf-8 -*-
"""
Created on Wed Jun  9 01:48:29 2021

Rizwan
"""


# Importing the libraries
#import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
# Importing the dataset
dataset = pd.read_csv('Salary_Data.csv')
X = dataset.iloc[:, :-1].values
y = dataset.iloc[:, 1].values

# Splitting the dataset into the Training set and Test set
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 1/3, random_state = 0)

# Fitting the simple linear rigression to train the set
from sklearn.linear_model import LinearRegression
regressor = LinearRegression()
regressor.fit(X_train, y_train)

## predicting The test data set results
#y_pred = regressor.predict(X_test)

#Visualization of prediction of the taining data set results
# =============================================================================
# plt.scatter(X_train, y_train, color = 'red')
# plt.plot(X_train, regressor.predict(X_train), color ='blue' )
# plt.title('Salary vs Experience')
# plt.xlabel('Years of experience')
# plt.ylabel('Salary')
# plt.show()
# =============================================================================

#Visualization of prediction of the testing data set results
plt.scatter(X_test, y_test, color = 'red')
plt.plot(X_test, regressor.predict(X_test), color ='blue' )
plt.title('Salary vs Experience')
plt.xlabel('Years of experience')
plt.ylabel('Salary')
plt.show()
