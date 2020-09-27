#!/usr/bin/env python
# coding: utf-8

# In[4]:


# package that are required for multiple linear regression.
# multiple linear regression :- It's a process of identifying the continous target variable with the 
# help of multiple predictor variable .
# x is our predictor variable and y is our target variable.
import numpy as np # for numerical operations.
import pandas as pd # for loading the dataset.
import matplotlib.pyplot as plt# for plot construction.


# In[9]:


mlr_df=pd.read_csv("D:\\data science\\dataset\\placement.csv")
# this line of command will help to retrive my dataset into my jupyter notebook or any other notebook.
# here:- mlr_df is my file name,pd is work with pandas,read_csv is command,the brackets represent the path of our data set.
print(mlr_df.shape)            # help in identifying the numbers of rows and column.
list(mlr_df)                   # help in identifying the name of the column.
                               # print is the command to print the output.


# In[14]:


x=mlr_df.iloc[:,0:3]  # x represent predictor variable,mlr_df is my datafile,.iloc is command and square
                      # bracket represent ROWS AND COLUMN
print(x)              # print information available in x
print(x.shape)        # print no. of rows and column in x only
print(list(x))        # columns name available in x 
print(x.ndim)         # dimension in x


# In[19]:


y=mlr_df['mba_p'] # represent my target variable
print(y)
print(y.shape)
print(y.ndim)


# In[ ]:





# In[20]:


x.corr() # co-relation :- relation btw objects but in mlr we need to check that their should not be any multicolinearty 
         # issue


# In[34]:


# dividing my data set into train and test, this can be done with the help of a package called sklearn we will import 
# train_test_split
#x_train,x_test,y_train,y_test

from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.8,random_state=1)
print(x_train.shape)
print(y_train.shape)


# In[35]:


#fitting the multiple linear regression to training set
from sklearn.linear_model import LinearRegression
regression=LinearRegression()
regression.fit(x_train,y_train)


# In[36]:


# prediction the test set result
y_predict=regression.predict(x_test)
print(y_predict)


# In[37]:


# check the r^2 value
from sklearn.metrics import r2_score
score=r2_score(y_test,y_predict)
print(score)


# In[38]:


# graph construction
plt.plot(y_test,y_predict)


# In[ ]:




