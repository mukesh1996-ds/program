import numpy as np

age = [25,26,45,35,36,54,22,80,65,46,55]
wgt = [60,65,70,96,77,75,71,81,87,90,95]
hgt = [5.6,5.1,5.4,5.9,6.1,6.0,5.8,7.1,5.3,6.2,5.7]
diabit = ["yes","no","yes","no","no","no","yes","yes","no","yes","yes"]

# assigning x and y variables

y = [] 

for v in diabit:
    if v == 'yes':
        y.append(1)
    else:
        y.append(0)
print(y)
print(type(y))
# converting into rows

y = np.c_[y]
print(" y avariables: ",y)


x = []

a = np.array(age)
# a = np.c_[a]
print("value of age is :",a)

w = np.array(wgt)
# w = np.c_[w]
print("values of weight :",w)

h = np.array(hgt)
# h = np.c_[h]
print("values of heights :",h)

# standivation formula

def sd(x):
    return (((x-x.mean())**2).sum()/(x.size-1))**0.5

# scaling
def scale(x):
    return (x-x.mean())/sd(x)

a = scale(a)
print("value of age \n",a)
w = scale(w)
print("value of weight \n",w)
h = scale(h)
print("value ofheight \n",h)

ones = np.ones(len(a))
x = np.c_[ones, a, w, h]
print(x)
print("shape of x",x.shape)

print("shape of y :",y.shape)

print(sd(a))
print(sd(w))
print(sd(h))

#=========================================================
# statical approach

from numpy.linalg import inv
betas = inv(x.T.dot(x)).dot(x.T.dot(y))
print("betas shape value are ",betas)
print(y)

# sigmodi function creations
def sigmoid(z):
    return 1 / (1 + np.exp(- z))

# derivative function creations
def derivative(df):
    return df *(1 - df)


yhat = sigmoid(x.dot(betas))
yhat[yhat>0.5] = 1
yhat[yhat<0.5] = 0
print("hat value of y",yhat)

from sklearn.metrics import accuracy_score
acc = accuracy_score(y,yhat)
print("accuracy score", acc )

#===================================================
# deeplearnig
# step 1
print(x.shape)
# my shape of x is (11,4) then my weight matrix will be 4 + 2 = 6
# step 2
print(y.shape)
# my shape of y is (11,1) my output value
# step 3 
np.random.seed(101)
w1 = 2 * np.random.random((4,6)) - 1
print("weight matrix for 1", w1)
print("shape of w1 :", w1)

np.random.seed(101)
w2 = 2 * np.random.random((6,1)) - 1
print("weight matrix for 2", w2)
print("shape of w2 :", w2)


l1 = sigmoid(x.dot(w1))
l2 = sigmoid(l1.dot(w2))
e2 = y - l2
d2 = e2 * derivative(l2)
e1 = d2.dot(w2.T)
d1 = e2 * derivative(l1)
w1a = x.T.dot(d1)
w2b = l1.T.dot(d2)

y_hat = sigmoid(l1.dot(w2b))

y_hat[y_hat > 0.5] = 1
y_hat[y_hat < 0.5] = 0
print(y_hat)

from sklearn.metrics import accuracy_score
acc = accuracy_score(y,y_hat)
print("accuracy score", acc )

#===============================================

# looping 

def loss(y,ycap):
    return ((y - ycap) **2).mean()



ploss = 0
flag = 0
convergence = 0.00000000001

for i in range (10000):
    l1 = sigmoid(x.dot(w1))
    l2 = sigmoid(l1.dot(w2))
    e2 = y - l2
    closs = loss(y,l2)
    diff = abs(closs - ploss)
    if diff <= convergence:
        print("training is completed",i+1,"iterations")
        flag = 1
        break
    if i % 100 == 0:
        print("current loss ", closs.round(4))
    d2 = e2 * derivative(l2)
    e1 = d2.dot(w2.T)
    d1 = e2 * derivative(l1)
    w1a = x.T.dot(d1)
    w2b = l1.T.dot(d2)
    ploss = closs
if flag == 0:
    print("training is not completed add few more iterations")

        
  















