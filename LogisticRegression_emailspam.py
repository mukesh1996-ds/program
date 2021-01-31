# Loading all the packages
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
# Importing the dataset 
df = pd.read_csv('spam.csv')
df.head()
# Preprocess the data 
from sklearn.preprocessing import LabelEncoder
lb = LabelEncoder()
df['type'] = lb.fit_transform(df['type'])
df['text'] = lb.fit_transform(df['text'])
df.head()
# string to numeric
df['type'] = pd.to_numeric(df['type'], errors = 'coerce')
df['text'] = pd.to_numeric(df['text'], errors = 'coerce')
df[['type']].shape
df[['type']].ndim
len(df[['type']])
df[['text']].shape
df[['type']].ndim
len(df[['text']])
# value sepration
x = df[['text']]
y = df[['type']]
# train_test_split
from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test = train_test_split(x,y, test_size = 0.2)
x_train.ndim
x_test.ndim
y_train.ndim
y_test.ndim
# implementing the Logistic Regression
from sklearn.linear_model import LogisticRegression
lr = LogisticRegression()
yhat = lr.predict(x_test)
score = lr.score(x_train,y_train)
# Model Evaluation
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix, accuracy_score
print("classification report of value", classification_report(y_test, yhat))
print("confusion matrix value is ",confusion_matrix(y_test, yhat))
print("accuracy value is ",accuracy_score(y_test, yhat))









