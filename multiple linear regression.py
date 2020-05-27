#!/usr/bin/env python
# coding: utf-8

# In[41]:


# packages that required
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd


# In[42]:


dataset=pd.read_csv("D:data science\\dataset\\Credit.csv")
print(dataset.shape)
print(list(dataset))


# In[43]:


# Putting my x&y 
x=dataset.iloc[:,1:8]
print(x)

# In[44]:


print(list(x))


# In[45]:


y=dataset.iloc[:,0]
print(y)

# In[46]:


print(list(y))
print(y.shape)


# In[47]:


# Label Encoder for dummy variable i.e One-Hot-Encoding
from sklearn import preprocessing
label_encoder = preprocessing.LabelEncoder()
x['Student']= label_encoder.fit_transform(x['Student'])
print(x.head())                                                  
                                                


# In[48]:


# spliting the dataset into train and test
from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.5,random_state=10)
print(x_train.shape)
print(y_train.shape)


# In[49]:


#fitting the multiple linear regression to training set
from sklearn.linear_model import LinearRegression
regression=LinearRegression()
regression.fit(x_train,y_train)


# In[56]:


# prediction the test set result
y_predict=regression.predict(x_test)
print(y_predict)


# In[55]:


# check the r^2 value
from sklearn.metrics import r2_score
score=r2_score(y_test,y_predict)
print(score)


# In[54]:


plt.plot(y_test,y_predict)


# In[ ]:




