import pandas as pd
df = pd.read_csv("E:\\data science\\dataset\\gini_index.csv")
print(df)


# preprocessing 
from sklearn import preprocessing
label_encoder = preprocessing.LabelEncoder()
df['Gender'] = label_encoder.fit_transform(df['Gender'])
print(df['Gender'])
df['Class'] = label_encoder.fit_transform(df['Class'] )
print(df['Class'] )
#df['Cricket']  = label_encoder.fit_transform(df['Cricket'])
#print(df['Cricket'])

print("list of elements in dataset",df)

# allocating x and y

x = df.iloc[:,0:1]
print("value of x",x)

y = df['Cricket']
print("value of y",y)


# Decision tree classifier
from sklearn.tree import DecisionTreeClassifier
dt= DecisionTreeClassifier()
result = dt.fit(x,y)
print("decision tree result",result)

y_pred = dt.predict(x)
print("predicted result",y_pred)

from sklearn import metrics
cm = metrics.confusion_matrix(y,y_pred)
print("confussion metrics ",cm)

accuracy = metrics.accuracy_score(y,y_pred)
print(accuracy)

# tree construction 

from sklearn import tree
import graphviz
dot = tree.export_graphviz(dt, filled = True)
graph = graphviz.Source(dot)
print(graph)
