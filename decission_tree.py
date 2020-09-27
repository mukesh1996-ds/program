import pandas as pd
df = pd.read_csv("E:\\data science\\dataset\\Diab.csv")
print(df.shape)
#====================================================
x = df.iloc[:, 0:4]
print(x.shape)
print(list(x))
y = df["Diabetic"]
print(list(y))
#==========================================
from sklearn.tree import DecisionTreeClassifier
dt = DecisionTreeClassifier()
result = dt.fit(x, y)
print("result of dt is", result)
y_pred = dt.predict(x)
print(" result of predicted value:",y_pred)
#=================================================
from sklearn import metrics
cm = metrics.confusion_matrix(y,y_pred)
print("Result of confussion martics is:",cm)
accuracy = metrics.accuracy_score(y,y_pred)
print("Result of accuracy score is:",accuracy)
#===========================================
from sklearn import tree
diagram = tree.plot_tree(dt)
print(diagram)


#===============================
