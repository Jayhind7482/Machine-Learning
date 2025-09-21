import numpy as np
import pandas as pd
from sklearn.datasets import make_blobs
from kmeans import KMeans
import matplotlib.pyplot as plt 
#print("hello world")

# centroid = [(-5,5) , (5,5) , (-2.5 , 2.5) , (2.5 , -2.5)]
# std = [1,1,1,1]

# X,y = make_blobs(centers = centroid , cluster_std=std , n_samples=100, n_features=2 ,random_state=3)
df = pd.read_csv("student_clustering.csv")
X = df.iloc[:,:].values

km = KMeans(max_iter = 2000 , n_cluster=4)
y_pred = km.fit_predict(X)
#print(y_pred)
plt.scatter(X[y_pred==0 ,0] ,X[y_pred==0,1] , c='r')
plt.scatter(X[y_pred==1 ,0] ,X[y_pred==1,1] , c='b')
plt.scatter(X[y_pred==2 ,0] ,X[y_pred==2,1] , c='y')
plt.scatter(X[y_pred==3 ,0] ,X[y_pred==3,1] , c='g')
plt.show()





