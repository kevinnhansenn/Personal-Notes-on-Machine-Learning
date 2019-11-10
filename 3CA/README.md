#	3.	Clustering Algorithm

## 	K-Means Clustering

![KNN](/3CA/Assets/KNN.png)

### The following are the steps of KNN:
-	Choose number of K clusters
-	Seelct random K point, center centroids 
-	Assign each data with its closest centroid
-	Compute and place new centroid of each cluster
-	Reassign each data point to the new closest centroid 
-	Repeat point 4 and 5 until finish

`KMeans -> is the library used`

```python
# Using the elbow method to find the optimal number of clusters
from sklearn.cluster import KMeans
wcss = []
for i in range(1, 11):
    kmeans = KMeans(n_clusters = i, init = 'k-means++', random_state = 42)
    kmeans.fit(X)
    wcss.append(kmeans.inertia_)
plt.plot(range(1, 11), wcss)
plt.title('The Elbow Method')
plt.xlabel('Number of clusters')
plt.ylabel('WCSS')
plt.show()

# Fitting K-Means to the dataset
kmeans = KMeans(n_clusters = 5, init = 'k-means++', random_state = 42)
y_kmeans = kmeans.fit_predict(X)
```

##	Hierarichal Clustering

![HC](/3CA/Assets/HC.png)

### There are two types of HC:
-	Agglomorative
-	Divisive

### Steps for Agglomorative:
-	Make each data point a single point cluster
-	Take two closest data points and make them one cluster
-	Take the two closest cluster and make them one cluster 
-	Repeat step 3 until there is only one cluster left 

`sch -> library used for hierarichal clustering`

```python
import scipy.cluster.hierarchy as sch
dendrogram = sch.dendrogram(sch.linkage(X, method = 'ward'))
plt.title('Dendrogram')
plt.xlabel('Customers')
plt.ylabel('Euclidean distances')
plt.show()

# Fitting Hierarchical Clustering to the dataset
from sklearn.cluster import AgglomerativeClustering
hc = AgglomerativeClustering(n_clusters = 5, affinity = 'euclidean', linkage = 'ward')
y_hc = hc.fit_predict(X)
```