# loading Required packages
import pandas as pd

# Loading data set

df= pd.read_csv('D:\\data science\\dataset\\Diab1.csv')
print(df.shape)

# X and Y variable
x=df.iloc[:,0:4]
print(x)
y=df['Diabetic']
print(y)

# Train and Test data

from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test=train_test_split(x,y,stratify=y)
print(x_train.shape)
print(x_test.shape)
print(y_train.shape)
print(y_test.shape)


# logical Regression
from sklearn.linear_model import LogisticRegression
lr=LogisticRegression()
model=lr.fit(x_train,y_train)
print(model)
y_pred=lr.predict(x_test)
print(y_pred)
print(y_test)

# confusion mrtics
cm=pd.crosstab(y_test,y_pred)
print(cm)

# loading all evaluation packages

from sklearn.metrics import accuracy_score
accurcy=accuracy_score(y_test,y_pred)
print("accoracy",accurcy)


# log loss entropy

from sklearn.metrics import log_loss
loss=log_loss(y_test,y_pred)
print("Logloss error",loss)


