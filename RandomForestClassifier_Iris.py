# Loading packages
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
#loading data set
df = pd.read_csv("Iris.csv")
df.head()
# label encoder
from sklearn.preprocessing import LabelEncoder
lb = LabelEncoder()
df['Species'] = lb.fit_transform(df['Species'])
# value seperation
x = df.iloc[:,1:5]
x.shape
x.size
x.ndim
y = df['Species']
y.size
y.shape
y.ndim
# spliting
x_train,x_test,y_train,y_test = train_test_split(x,y,test_size = 0.3, random_state = 0)      
print(x_train.shape, x_train.ndim)
print(y_train.shape, y_train.ndim)
print(x_test.shape, x_test.ndim)
print(y_test.shape, y_test.ndim)
# loading the model
model = RandomForestClassifier(n_estimators = 100)
model.fit(x_train,y_train)
y_pred = model.predict(x_test)
plt.scatter(x_test.iloc[:,1],y_test)
plt.plot(y_test,y_pred)
plt.legend()
plt.show()