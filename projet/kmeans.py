from dataset import digits_dataset
import numpy as np
from sklearn.metrics import pairwise_distances_argmin, silhouette_score
class Kmeans:
    def __init__(self, k=10, tol = 1e-4, max_iter=1000, random_state=42):
        self.data = digits_dataset.data
        self.k = k
        self.max_iter = max_iter
        self.random_state = random_state
        self.tol = tol
        self.centroids = None
        self.clusters = None

    def __str__(self):
        if self.centroids is not None and self.clusters is not None:
            return f"""Kmeans with {self.k} clusters
Centroids: {self.centroids.shape}
Clusters: {len(self.clusters)} elements"""
        else:
            return "Kmeans object not initialized"
    
    def initialize_centroids(self):
        np.random.seed(self.random_state)
        indices = np.random.permutation(len(self.data))[:self.k]
        self.centroids = self.data[indices]

    def assign_clusters(self):
        self.clusters = pairwise_distances_argmin(self.data, self.centroids)
    
    def update_centroids(self):
        new_centroids = np.array([self.data[self.clusters == i].mean(axis=0) for i in range(self.k)])
        return new_centroids
        
    def fit(self):
        self.initialize_centroids()
        for i in range(self.max_iter):
            old_centroids = self.centroids
            self.assign_clusters()
            self.centroids = self.update_centroids()
            if np.linalg.norm(self.centroids - old_centroids) < self.tol:
                print(f"Convergence reached after {i+1} iterations.")
                break

    def predict(self, test_data):
        return pairwise_distances_argmin(test_data, self.centroids)

    def score(self, test_data):
        cluster_labels = self.predict(test_data)
        if len(np.unique(cluster_labels)) > 1:
            return silhouette_score(test_data, cluster_labels)
        else:
            return -1 
    
kmeans = Kmeans()
kmeans.fit()
print(kmeans.centroids)
print(kmeans.clusters)