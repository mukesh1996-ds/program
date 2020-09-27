# pandas are used for loading the dataset into the environment
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
df = pd.read_csv("D:\\data science\\dataset\\MARK LIST.csv")
print(df)
print(list(df))
print(df.shape)
print(df.head)
print(df.describe())
print(df.nunique())
print(df.isnull().sum())
corelation = df.corr()
print("corelation value are /n ", corelation)
sns.heatmap(corelation,xticklabels=corelation.columns, yticklabels=corelation.columns, annot=True)
sns.pairplot(df)
sns.relplot(x='hindi', y='english', hue='result',data=df)
sns.distplot(df['hindi'], bins=5)
sns.catplot(x='hindi',kind='box',data=df)
sns.catplot(x='english',kind='box',data=df)
sns.catplot(x='maths',kind='box',data=df)
sns.catplot(x='science',kind='box',data=df)
sns.catplot(x='social',kind='box',data=df)
plt.show()








