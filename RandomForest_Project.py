# loading the packages
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
#loading dataset
df = pd.read_csv('project.csv')
df.head()
# divide the data set
x= df.iloc[:,0:7]
x.ndim
x.size
x.shape
y = df.iloc[:,:7]
y.size
y.shape
y.ndim
#train_test_
from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test = train_test_split(x,y, test_size = 0.2, random_state = 42 )
print("x_train", x_train.shape, x_train.ndim, x_train.size)
print("y_train", y_train.shape, y_test.ndim, y_train.size)
print("x_test", x_test.shape, x_test.ndim, x_test.size)
print("y_test", y_test.shape, y_test.ndim, y_test.size)
# Loading the model
from sklearn.ensemble import RandomForestRegressor
model = RandomForestRegressor()
model.fit(x_train,y_train)
y_predict = model.predict(x_test)
# evaluation
from sklearn import metrics
print("r^2", metrics.r2_score(y_test, y_predict))
print("MSE", metrics.mean_squared_error(y_test,y_predict))
x_train.size
y_train.size
x_test.size
y_test.size
x_test1 = np.reshape(x_test,(4,7)) 
plt.scatter(x_test1,y_test)
np.arange(20)
plt.plot(x_test1,y_predict)







