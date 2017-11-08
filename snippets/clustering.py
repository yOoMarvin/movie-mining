
# BASIC KMEANS
from sklearn.cluster import KMeans
estimator = KMeans(n_clusters = 2)
labels = estimator.fit_predict(customer_data[['ItemsBought', 'ItemsReturned']])
print(labels)
## or
estimator.fit(customer_data[['ItemsBought', 'ItemsReturned']])
print(estimator.labels_)




# PLOTTING CLUSTERS
import matplotlib.pyplot as plt
plt.title("KMeans #cluster = 2")
plt.xlabel('ItemsBought')
plt.ylabel('ItemsReturned')
plt.scatter(customer_data['ItemsBought'], customer_data['ItemsReturned'], c=estimator.labels_)
plt.show()


# SILHOUETTE SCORE
from sklearn.metrics import silhouette_score
silhouette = silhouette_score( customer_data[['ItemsBought', 'ItemsReturned']], labels)
silhouette


# DBSCAN
from sklearn.cluster import DBSCAN
db = DBSCAN().fit(customer_data[['ItemsBought', 'ItemsReturned']])
plt.scatter(customer_data['ItemsBought'], customer_data['ItemsReturned'], c=db.labels_)
plt.show()



#DENDOGRAM
from scipy.cluster.hierarchy import dendrogram, linkage

Z = linkage(customer_data[['ItemsBought', 'ItemsReturned']], 'ward')

plt.title('Hierarchical Clustering Dendrogram')
plt.xlabel('Customer IDs')
plt.ylabel('distance')
dendrogram(Z, labels=customer_data['Customer ID'].values)
plt.show()

## truncate

plt.title('Dendrogram - 3 clusters')
plt.xlabel('Count of Customers')
plt.ylabel('distance')
dendrogram(Z,
        truncate_mode='lastp',
        p=3)
plt.show()
