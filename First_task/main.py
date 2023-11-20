import pandas as pd
import seaborn as sns
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn import preprocessing


sns.set()

raw_data = pd.read_csv('./Mall_Customers.csv')

print(raw_data.head())
print("----------------description---------------")
print(raw_data.describe())

data = raw_data.copy()

data['Gender'] = data['Gender'].map({'Male': 0, 'Female': 1})

print(data.isnull().sum())

features = data[['Gender', 'Age', 'Annual Income (k$)', 'Spending Score (1-100)']].values

featuresScaled = preprocessing.scale(features)

# Elbow Method
WCSS = []
for i in range(1, 10):
    kmeans = KMeans(n_clusters=i)
    kmeans.fit(featuresScaled)
    WCSS.append(kmeans.inertia_)

plt.plot(range(1, 10), WCSS, marker='o', linestyle='--')
plt.title('Elbow Method')
plt.xlabel('Number of Clusters')
plt.ylabel('WCSS')
plt.show()

numOfClusters = 5

kmeans = KMeans(numOfClusters)
clusters = kmeans.fit_predict(featuresScaled)

data['Cluster'] = clusters

clusterSummary = data.groupby('Cluster').mean()
print(clusterSummary)

plt.scatter(data['Annual Income (k$)'], data['Spending Score (1-100)'], c=data['Cluster'], cmap='rainbow')
plt.xlabel('Annual Income (k$)')
plt.ylabel('Spending Score (1-100)')
plt.title('Customer Segmentation')
plt.show()
