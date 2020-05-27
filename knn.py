# K NEAREST NEIGHBOUR:-
# It's a classification technique which is completely based on supervised learning.
# In KNN target variable should be discrete & also help to predict the nearest element in the dataset.
# completely based on the distance btw one data and other data.
# for distance calculation it will use eculidian distance.

# loading required packages.
import pandas as pd
import matplotlib.pyplot as plt
# loading of dataset
knn_df=pd.read_csv("D:\\data science\\dataset\\Housing_loan.csv")
# once my data set is loaded then need to select my x & y variables
x=knn_df.iloc[:,1:6]
print(x)
print(x.shape)
print(list(x))
y=knn_df["Loan_sanctioned"]
print(y)
print(y.shape)
print(list(y))
print(pd.crosstab(y,y))
#train & test dataset
from sklearn.model_selection import train_test_split # package for train and test
x_train,x_test,y_train,y_test=train_test_split(x,y,stratify=y,random_state=1)
print(x_train.shape)
print(y_train.shape)
print(x_test.shape)
print(y_test.shape)
print(pd.crosstab(y_test,y_test)) # cross chack the data
print(knn_df.groupby("Loan_sanctioned").size())
# loading knn package
from sklearn.neighbors import KNeighborsClassifier # package in which knn is present.
knn=KNeighborsClassifier(n_neighbors=5) # help to chack all the other attribute.
print("fitted value",knn.fit(x_train,y_train))
y_pred=knn.predict(x_test)
print("tested value",y_pred)
#loading confusion metrics 
from sklearn import metrics   
cm=metrics.confusion_matrix(y_test,y_pred)
print("confusion matrics",cm)
accuracy=metrics.accuracy_score(y_test,y_pred)
print("accuracy value",accuracy)
test_accuracy=[]
k1=range(1,21)
for i in k1:
    knn=KNeighborsClassifier(n_neighbors=i) # help to chack all the other attribute.
    print("fitted value",knn.fit(x_train,y_train))
    y_pred=knn.predict(x_test)
    print("tested value",y_pred)
    test_accuracy.append(metrics.accuracy_score(y_test,y_pred).round(2))
print(test_accuracy)
plt.plot(k1,test_accuracy,label="test accuracy")
plt.ylabel("accuracy")
plt.xlabel("k-values")
plt.legend()
plt.show()    


##########################################################################
ram_state=range(1,51)
k= range(1,51)
import numpy as np
acc=np.zeros(shape=(51,51),dtype=float)
for a in ram_state:
    for b in k:
        x_train,x_test,y_train,y_test=train_test_split(x,y,stratify=y,random_state=a)
        knn=KNeighborsClassifier(n_neighbors=b) # help to chack all the other attribute.
        print("fitted value",knn.fit(x_train,y_train))
        acc[a][b]=knn.score(x_test,y_test)
        
print(acc)

df_accuracy=pd.dataframe(acc)
print(df_accuracy)
#################################################################################
 















