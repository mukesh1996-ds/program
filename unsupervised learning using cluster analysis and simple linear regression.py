# UNSUPERVISED LEARNING
# In this their is no target variable, we have to identify the target variable and the preform supervised learning for predictions.
# Let's work with "CLUSTER ANALYSIS"
# In cluster analysis we are going to create a target variable with the help of scatter plot.
# In cluster analysis we will divide our data into different cluster according to their similarties.
# Their are basically two types of method for cluster analysis :-
#   * AGGLOMERTIVE CLUSTERING :- They have multiple linkage method 
#                                -single linkage,complete linkage,avreage linkage,centroide linkage,median linkage,ward linkage
#   * K-MEAN CLUSTERING
# Loading required packages for AGGLOMERTIVE CLUSTERING

import numpy as np  # for working with numerical values
import pandas as pd # for loading the data set
import matplotlib.pyplot as plt # for data visualization

ul=pd.read_csv("D:\data science\dataset\\orsales.csv")
print(ul.shape)
print(list(ul))
print(ul.ndim)

# Working with two dimensional array
t=ul.iloc[:,3:]
print(t)

# for cluster we need to upload some package i.e
import scipy.cluster.hierarchy as shc
plt.figure(figsize=(10,7))
plt.title("information")
shc.dendrogram(shc.linkage(t,method="single"))

from sklearn.cluster import AgglomerativeClustering
cluster = AgglomerativeClustering(n_clusters=4, affinity="euclidean", linkage="complete")
cluster.fit_predict(t)
y=cluster.labels_
print(y)
plt.figure(figsize=(10,7))
plt.scatter(t.iloc[:,0],t.iloc[:,1],c=cluster.labels_,cmap="rainbow")
plt.xlabel("Total_Retail_Price")
plt.ylabel(" Profit ")
plt.show()

# seperate x AND y variable
x=t.iloc[:,0]
x=x[:,np.newaxis]
print(x)

x.ndim

y=t.iloc[:,1]
print(y)
y.ndim


# working with simple linear regression
from sklearn.linear_model import LinearRegression
regression=LinearRegression()
regression.fit(x,y)
print(regression.intercept_) # for finding out b0 value
print(regression.coef_) # for finding out b1 values 
y_pred=regression.predict(x) # for prediction
print(y_pred)


# calculated MSE
from sklearn.metrics import mean_squared_error
mse=mean_squared_error(y,y_pred)
print(mse)

# for adding constant value
import statsmodels.api as sm
x=sm.add_constant(x)
print(x)

# calculation r2 value
import statsmodels.regression.linear_model as sm2
value=sm2.OLS(y,x).fit()
print(value.summary())



