# -*- coding: utf-8 -*-
"""
Created on Sat Aug 24 18:02:40 2019

@author: Raghuram
"""

# importing the libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# importing the dataset
dataset = pd.read_csv("Salary_Data.csv")
X = dataset.iloc[:, :-1].values
y = dataset.iloc[:,-1]

# splitting the dataset into the Training set and Test set
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 1/3, random_state = 29)

# Linear Regression model
from sklearn.linear_model import LinearRegression
reg = LinearRegression()
reg.fit(X_train,y_train)

#Predicting the test results
y_pred = reg.predict(X_test)

#Visualization
plt.scatter(X_train, y_train, color = 'red')
plt.plot(X_train, reg.predict(X_train), color = 'blue')
plt.title("Experience Vs Salary")
plt.xlabel("Years of Experience")
plt.ylabel("Salaries")
plt.show()


plt.scatter(X_test, y_test, color = 'red')
plt.plot(X_train, reg.predict(X_train), color = 'blue')
plt.title("Experience Vs Salary")
plt.xlabel("Years of Experience")
plt.ylabel("Salaries")
plt.show()

