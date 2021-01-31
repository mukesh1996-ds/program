# Loading all the packages
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
# Loding the data set
df = pd.read_csv('Housing_loan.csv')
df.head()
# Value Seperation
x = df.iloc[:,0:5]
x.ndim
y = df.iloc[:,5]
# train and test 
from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test = train_test_split(x, y, test_size = 0.2, random_state = 0)
# importing logistic regression
from sklearn.linear_model import LogisticRegression 
LR = LogisticRegression()
# fit the model
LR.fit(x_train, y_train)
yhat = LR.predict(x_test)
score =LR.score(x_train,y_train)
# Model Evaluation
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix, accuracy_score
print("classification report of value", classification_report(y_test, yhat))
print("confusion matrix value is ",confusion_matrix(y_test, yhat))
print("accuracy value is ",accuracy_score(y_test, yhat))
# visuatization
sns.heatmap(pd.DataFrame(confusion_matrix(y_test, yhat)))
plt.scatter(x_test.iloc[:,:1],y_test)
plt.scatter(x_test['Experience'],y_test)
plt.scatter(x_test['Income'],y_test)
plt.scatter(x_test['Family'],y_test)
plt.scatter(x_test['Education'],y_test)
plt.plot(x_test, yhat)
df.head()
result = LR.predict(1,0)

