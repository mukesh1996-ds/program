import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

df = pd.read_csv('Placement_Data_Full_Class.csv')
df.head()
df.isnull().sum()
sns.heatmap(df.isnull(), yticklabels = False)
from sklearn.preprocessing import LabelEncoder
lb = LabelEncoder()
df['workex'] = lb.fit_transform(df['workex'])
df['status'] = lb.fit_transform(df['status'])
df['gender'] = lb.fit_transform(df['gender'])
df

x = df.iloc[:,0:8]
y = df['status']
from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x,y, test_size = 0.2 )
from sklearn.linear_model import LogisticRegression
LR = LogisticRegression()
LR.fit(x_train,y_train)
y_pred = LR.predict(x_test)
from sklearn.metrics import confusion_matrix, accuracy_score
cm = confusion_matrix(y_test,y_pred)
accracy = accuracy_score(y_test, y_pred)
y_test = np.array(y_test)
plt.scatter(x = 'y_test', y = 'y_pred')
plt.show()