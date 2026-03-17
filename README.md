# Implementation of Random Forest Algorithm for Weather Prediction
## AIM:
To write a program to predict daily temperature , PM2.5 pollution level and Energy based on environmental sensor data using Random Forest Algorithm.

## Problem Statement and Dataset



## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm
1.Load the customer dataset and select the relevant features such as Annual Income and Spending Score.
2. Choose the number of clusters K and initialize K centroids randomly.
3. Assign each data point to the nearest centroid using Euclidean distance and update the centroids by calculating the mean of each cluster. 
4. Repeat Step 3 until the centroids no longer change and display the final clusters for customer segmentation.

Program:. 


## Program:
```
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
​
data = pd.read_csv("C:/Users/acer/Downloads/Mall_Customers.csv")
print(data.head())
​
X = data.iloc[:, [3, 4]].values
​
wcss = []
for i in range(1, 11):
    kmeans = KMeans(n_clusters=i, init='k-means++', random_state=42)
    kmeans.fit(X)
    wcss.append(kmeans.inertia_)
​
plt.figure(figsize=(8,5))
plt.plot(range(1, 11), wcss, marker='o')
plt.title('Elbow Method')
plt.xlabel('Number of Clusters')
plt.ylabel('WCSS')
plt.show()
​
kmeans = KMeans(n_clusters=5, init='k-means++', random_state=42)
y_kmeans = kmeans.fit_predict(X)
plt.figure(figsize=(8,6))
plt.scatter(X[y_kmeans == 0, 0], X[y_kmeans == 0, 1], s=100, c='red', label='Cluster 1')
plt.scatter(X[y_kmeans == 1, 0], X[y_kmeans == 1, 1], s=100, c='blue', label='Cluster 2')
plt.scatter(X[y_kmeans == 2, 0], X[y_kmeans == 2, 1], s=100, c='green', label='Cluster 3')
plt.scatter(X[y_kmeans == 3, 0], X[y_kmeans == 3, 1], s=100, c='cyan', label='Cluster 4')
plt.scatter(X[y_kmeans == 4, 0], X[y_kmeans == 4, 1], s=100, c='magenta', label='Cluster 5')
​
plt.scatter(kmeans.cluster_centers_[:,0], 
            kmeans.cluster_centers_[:,1], 
            s=300, c='yellow', label='Centroids')
​
plt.title('Customer Segmentation using K-Means')
plt.xlabel('Annual Income (k$)')
plt.ylabel('Spending Score (1-100)')
plt.legend()
plt.show()
```

## Output:
<img width="1036" height="217" alt="image" src="https://github.com/user-attachments/assets/56699909-38bc-42c2-ba49-248bd41fc5b5" />
<img width="1046" height="680" alt="image" src="https://github.com/user-attachments/assets/5eb18394-348c-4ca3-841d-579649c7b927" />
<img width="1049" height="779" alt="image" src="https://github.com/user-attachments/assets/57be75a9-4570-4431-b3c5-53510ecfb93d" />


## Result:
