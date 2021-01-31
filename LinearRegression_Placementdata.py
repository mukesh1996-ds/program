# loading the packages
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn import linear_model
from sklearn import metrics
#===================================================
df = pd.read_csv('D:\\data science\\dataset\\placement.csv')
df.head()
df.isnull().sum()
#==================================================
x = df.iloc[:,0:3]
x.ndim
y = df.iloc[:,:-1]
y.ndim
#===================================================
x_train, x_test, y_train, y_test = train_test_split(x,y,test_size = 0.2, random_state = 42)

print(x_train.shape)
y_train = y_train[:,np.newaxis]
y_train.ndim
print(y_train.shape)

print(x_test.shape)
x_test.size
y_test.size
y_test = y_test[:,np.newaxis]
print(y_test.shape)

#===================================================
model = linear_model.LinearRegression()
model.fit(x_train, y_train)
predict = model.predict(x_test)
model.get_params()
print('Co-efficient of linear regression',model.coef_)
print('Intercept of linear regression model',model.intercept_)
print('Model R^2 Square value', metrics.r2_score(y_test, predict))
print('Mean squared error ', metrics.mean_squared_error(y_test, predict))
plt.scatter(x_test, y_test)
plt.plot(x_test, predict, color='red')
plt.xlabel('student marks')
plt.ylabel('adminision')
plt.title('Linear Regression')










