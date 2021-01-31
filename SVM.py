# Loading packages
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.svm import SVR
#loading data set
df = pd.read_csv("placement.csv")
df.head()
# Seperating of x and y 
x = df.iloc[:,0:3]
x.shape
x.ndim
x.size
y = df.iloc[:,3]
y = y[:,np.newaxis]
y.shape
y.size
y.ndim
# ploting
plt.plot(x)
plt.legend(x)
# spliting
x_train,x_test,y_train,y_test = train_test_split(x,y,test_size = 0.22, random_state = 0)      
print(x_train.shape, x_train.ndim, x_train.size)
print(x_test.shape, x_test.ndim, x_test.size)
print(y_train.shape, y_train.ndim, y_train.size)
print(y_test.shape, y_test.ndim, y_test.size)
# loading the model
model = SVR(kernel = 'rbf')
model.fit(x_train,y_train)
y_pred = model.predict(x_test)
y_test = np.resize(y_test, (48,3))
plt.scatter(x_test,y_test)
y_pred = y_pred[:,np.newaxis]
plt.plot(x_test,y_pred)
plt.legend()
plt.show()
























