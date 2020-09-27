import pandas as pd
df = pd.read_csv("E:\\data science\\dataset\\taxi.csv")
print(df)

# selecting the required term

df = df.iloc[:,3:]
print(df)

import scipy.cluster.hierarchy as sch
import matplotlib.pyplot as plt

plt.figure(figsize= (10,7))
plt.title("parking information")
sch.dendrogram(sch.linkage(df, method = "complete"))

from sklearn.cluster import AgglomerativeClustering as ag
cluster = ag(n_clusters = 4 , affinity = 'euclidean', linkage='complete')
cluster.fit_predict(df)         
y = cluster.labels_
print(y)


plt.figure(figsize=(10,7))
plt.scatter(df.iloc[:,0],df.iloc[:,1],c = cluster.labels_, cmap = "rainbow")
plt.show()


from sklearn.linear_model import LinearRegression
lm = LinearRegression().fit(df,y)
print(lm.intercept_)
print(lm.coef_)
Y_pred = lm.predict(df)
Y_pred
y - Y_pred
# Mean square error
from sklearn.metrics import mean_squared_error
MSE = mean_squared_error(y,Y_pred)
print("error value",MSE)








