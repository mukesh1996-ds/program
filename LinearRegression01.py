# loading the packages
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn import linear_model
from sklearn import metrics
#===================================================
df = pd.read_csv('D:\\data science\\dataset\\ta.csv')
df.head()
df.isnull().sum()
#===================================================
np.arange(26)
plt.hist(df['Monthlyincome'], data = 'df')
plt.show()
#=====================================================
# data visualization
sns.jointplot(x = df['Monthlyincome'], y = df['Numberofweeklyriders'])
#=========================================================================
# splitting the dataset
x_train, x_test, y_train, y_test = train_test_split(df['Monthlyincome'],
                                                     df['Numberofweeklyriders'],test_size = 0.2, random_state = 42)

x_train = x_train[:,np.newaxis]
x_train.ndim
print(x_train.shape)

y_train = y_train[:,np.newaxis]
y_train.ndim
print(y_train.shape)
x_test = x_test[:,np.newaxis]
x_test.ndim
print(x_test.shape)
y_test = y_test[:,np.newaxis]
y_test.ndim
print(y_test.shape)

#===================================================
cls = linear_model.LinearRegression()
cls.fit(x_train, y_train)
predict = cls.predict(x_test)
cls.get_params()
print('Co-efficient of linear regression',cls.coef_)
print('Intercept of linear regression model',cls.intercept_)
print('Model R^2 Square value', metrics.r2_score(y_test, predict))
print('Mean squared error ', metrics.mean_squared_error(y_test, predict))
plt.scatter(x_test, y_test)
plt.plot(x_test, predict, color='red', linewidth=3)
plt.xlabel('Hours')
plt.ylabel('Marks')
plt.title('Linear Regression')