#!/usr/bin/env python
# coding: utf-8

# In[ ]:





# # Loading the data set 

# In[15]:


df = open('E:\\data science\\dataset\\text.txt')
df


# 
# #  reading the data

# In[16]:


lines = df.readlines()
lines


# # seperating of x and y variable

# In[17]:


text = []
label = []


# # assign values

# In[18]:


for line in lines:
    w = line.lower().strip().split(',')
    t =w[0]
    l = 1
    if w[1] == 'neg':
        l =0
    text.append(t)
    label.append(l)


# # printing the seperate values

# In[19]:


print(text)
print(label)


# # converting value to numeric

# In[27]:


# count vectorizer
from sklearn.feature_extraction.text import CountVectorizer
vec = CountVectorizer()
vec.fit(text)
#transform
v = vec.transform(text).toarray()
print(v)
print(v.shape)


# # term freq - inverse document frequency

# In[32]:


from sklearn.feature_extraction.text import TfidfVectorizer
vec1 = TfidfVectorizer()
vec1.fit(text)
x = vec1.transform(text).toarray()

print(x)

y = label
y


# # apply any of the learning

# In[34]:


from sklearn.tree import DecisionTreeClassifier
model = DecisionTreeClassifier()
model.fit(x,y)
yhat = model.predict(x)
yhat


# # Accuracy Score

# In[37]:


from sklearn.metrics import accuracy_score
acc = accuracy_score(yhat,y)
acc


# In[ ]:





# In[ ]:





# In[ ]:




