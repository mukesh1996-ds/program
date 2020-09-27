import numpy as np
import pandas as pd
df = pd.read_csv("D:\\data science\\dataset\\Diab.csv")
print(df)
print(list(df))

# seperate x and y value
x = df.iloc[:,0:3]
print("list of x ", list(x))

y = df['Diabetic']
print("list of y",list(y))

# function creation
# standerd divation
def sd(x):
   return (((x-x.mean())**2).sum()/(x.size-1))**0.5

# standization
def scale(x):
    return (x - x.mean())/sd(x)

# sigmoid
def sigmoid(z):
    return 1 / (1 + np.exp(-z))

# derivatiove
def derivative(ycap):
    return ycap * ( 1 - ycap)

# logloss
def loss(y,ycap):
    return((y - ycap) ** 2).mean()

# applying function
print("Standivation is ",sd(x))

ones = np.ones(len(x))
x = np.c_[ones,x]
x.shape
print(x)



# statistical approach
from numpy.linalg import inv
betas = inv(x.T.dot(x)).dot(x.T.dot(y))
print("Betas value:",betas.shape)
print("Shape of x ",x.shape)
print("value of y",y)

yc = sigmoid(x.dot(betas))
y

yc[yc>0.5] = 1
yc[yc<0.5] = 0
print(yc)


# accuracy check 
from sklearn.metrics import accuracy_score
acc = accuracy_score(y,yc)
print(acc)

# by checking the accuracy we got the 0.35 accuracy score whice is not good

# Deep learning approach ===========================================

#input matrix =20 x 5
#w1= (20 x 5).dot(5 x 8)=20 x 8
#l1=sigmoid( 20 x 8)
#w2= (8 x 1)
#l2= sigmoid (20 x 1)
#yhat = 20 x 1
#y
#e2 = y-yhat
# #d2 = e2 * [y hat *(1-yhat)]
#   = 20 x 1
#e1= error.dot(w2.t)
#  = (20 x 1).dot (1 x 8)= 20 x 8
#d1= e1 * [y hat *(1- yhay)]
#  = 20 x 8
#w1+ = X.t.dot (d1)
#    = (5 x 20).(20 x 8)= 5 x 8
#w2+ = l1.t.dot(d2)
#   =(8 x 20).(20 x 1)= 8 x 1


#setp1
print(x.shape)
#setp2
print(y.shape)
#setp3
np.random.seed(101)
W1 = 2 * np.random.random((5,6)) - 1
print(W1)
print("List of W1",list(W1))
print("The shape of w1 is ",W1.shape)
np.random.seed(101)
W2 = 2 * np.random.random((6,1)) - 1
print("value of w2",W2)
print("shape of w2",W2.shape)


L1 = sigmoid(20,6)
L2 = sigmoid(20,1)  # y hat

e2 = y - L2
D2 = e2 * derivative(L2)
e1 = D2.dot(W2.T)
D1 = e1 * derivative(L1)
W1a = x.T.dot(D1)
W2b = L1.T.dot(D2)




yhat = sigmoid(L2.dot(W2b))

L2[L2>0.5] = 1
L2[L2<0.5] = 0
L2

from sklearn.metrics import accuracy_score
accuracy_score(y,L2).round(2)
'''
ploss = 0
flag = 0
convergence = 0.0000000001

for i in range(1000000):
    L1 = sigmoid(x.dot(W1))
    print("shape of l1 is",L1.shape)
    print("shape of w1 is", W1.shape)
    L2 = sigmoid(L1.dot(W2))  # yhat
    print("shape of l2",L2)
    e2 = y - L2
    closs = loss(y, L2)
    diff = abs(closs - ploss)
    if diff <= convergence:
        print("Training is completed", i + 1, "iterations")
        flag = 1
        break
    if i % 1000 == 0:
        print("current loss", closs.round(15))
    D2 = e2 * derivative(L2)
    e1 = D2.dot(W2.T)
    D1 = e1 * derivative(L1)
    W1 += x.T.dot(D1)
    W2 += L1.T.dot(D2)
    ploss = closs

if flag == 0:
    print("Training is not completed add few more interations")

L1.shape
W2.shape
yhat = sigmoid(L1.dot(W2))

yhat[yhat > 0.5] = 1
yhat[yhat < 0.5] = 0
print("y hat value are",yhat)

from sklearn.metrics import accuracy_score

acc_score = accuracy_score(Y, yhat).round(2)
print("Score value is",acc_score)
'''