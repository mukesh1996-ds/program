# Loading packages
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
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
x_train,x_test,y_train,y_test = train_test_split(x,y,test_size = 0.2, random_state = 0)      
print(x_train.shape, x_train.ndim, x_train.size)
print(y_train.shape, y_train.ndim, y_train.size)
print(x_test.shape, x_test.ndim, x_test.size)
print(y_test.shape, y_test.ndim, y_test.size)
#y_test = np.random.rand(30,4)
#y_test.size
# loading the model
model = SVC(kernel = 'rbf')
model.fit(x_train,y_train)
y_pred = model.predict(x_test)
plt.scatter(x_test.iloc[:,:1],y_test)
plot = plt.plot(y_test,y_pred)
plt.legend(plot)
plt.show()