# -*- coding: utf-8 -*-
"""
Created on Thu Dec  3 11:51:40 2020

@author: mksmu
"""

# Loading the packages
import pandas as pd
import numpy as np
#====================
# Laoding the dataset
df = pd.read_csv('IceCreamData.csv')
df.head()
df.isnull().sum()
#==================
# Loading train_test_split Packages
from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test = train_test_split(df["Temperature"], df["Revenue"], test_size = 0.2)
x_train = x_train[:,np.newaxis]
print(x_train.ndim)
print(x_train.shape)
y_train = y_train[:,np.newaxis]
print(y_train.ndim)
print('y_train shape',y_train.shape)
x_test = x_test[:,np.newaxis]
print(x_test.ndim)
print('x_test shape',x_test.shape)
y_test = y_test[:,np.newaxis]
print(y_test.ndim)
print('y_test shape',y_test.shape)

#===================
# fitting the random forest regressor 
from sklearn.ensemble import RandomForestRegressor
model_regressor = RandomForestRegressor(n_estimators = 10, random_state = 0)
model_regressor.fit(x_train,y_train)
y_pred = model_regressor.predict(x_test)
#=====================
# comparing real values with predicted values
newdf = pd.DataFrame({'original value':y_test.reshape(-1),'predicted value': y_pred.reshape(-1)})
newdf
#=========
# evaluation
from sklearn import metrics
print("r^2 value ",metrics.r2_score(y_test,y_pred))
print("MSE", metrics.mean_squared_error(y_test,y_pred))
import matplotlib.pyplot as plt
plt.scatter(x_test.size,y_test.size)
plt.plot(x_test,y_pred)
plt.xlabel('ride')
plt.ylabel('money')
plt.title('RANDOM FOREST')
