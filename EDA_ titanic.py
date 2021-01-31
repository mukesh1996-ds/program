# import packages required
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
#==============================
# Loading datasets
train = pd.read_csv('D:\\data science\\dataset\\train.csv')
train.head()
#=============================
train.isnull().sum()
#=============================
sns.heatmap(train.isnull(), yticklabels = False)
#==============================
sns.set_style('whitegrid')
sns.countplot(x = 'Survived', data = train)

sns.set_style('whitegrid')
sns.countplot(x = 'Survived',hue = 'Sex', data = train)

sns.set_style('whitegrid')
sns.countplot(x = 'Survived', hue = 'Pclass', data = train)
#=============================
# null value handle
sns.distplot(train['Age'].dropna(), bins = 40)
sns.countplot(x = 'SibSp', data = train)
train['Fare'].hist(bins = 40)
#=========================
plt.figure(figsize = (12,7))
sns.boxplot(x = 'Pclass', y = 'Age', data = train, color = 'g')

def impute_age (cols):
    Age = cols[0]
    Pclass = cols[1]
    if pd.isnull(Age):
        if Pclass == 1:
            return  37
        elif Pclass == 2:
            return  24
        else:
            return 24
    else:
        return Age
        
train['Age'] = train[['Age','Pclass']].apply(impute_age,axis = 1)
sns.heatmap(train.isnull(), yticklabels = False)
train.drop('Cabin', axis = 1, inplace = True)
train.head()
#=========================
pd.get_dummies(train['Embarked'], drop_first = True).head()
sex = pd.get_dummies(train['Sex'], drop_first = True)
embarked = pd.get_dummies(train['Embarked'], drop_first = True)
train.head()
train.drop(['PassengerId','Sex', 'Embarked', 'Name', 'Ticket'], axis = 1, inplace = True)
train.head()
train = pd.concat([train,sex,embarked], axis = 1)
train.head()
#=============================
x = train.iloc[:,1:8]
x.ndim
y = train['Survived']
y.ndim

#=============================
from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x,y, test_size = 0.3 )
from sklearn.linear_model import LogisticRegression
lr = LogisticRegression()
lr.fit(x_train,y_train)
ypred = lr.predict(x_test)
from sklearn.metrics import confusion_matrix
acc = confusion_matrix(y_test, ypred)
from sklearn.metrics import accuracy_score
score = accuracy_score(y_test, ypred)
score