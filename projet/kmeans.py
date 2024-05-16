from dataset import digits_dataset
import numpy as np
from sklearn.metrics import pairwise_distances_argmin, silhouette_score, completeness_score

class Kmeans:
    def __init__(self, k=10, tol=1e-4, max_iter=1000, random_state=42):
        self.data = digits_dataset.data
        self.target = digits_dataset.target
        self.k = k
        self.max_iter = max_iter
        self.random_state = random_state
        self.tol = tol
        self.centroids = None
        self.clusters = None
        self.cluster_to_label = None

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
    
    def silhouette(self):
        if len(np.unique(self.clusters)) > 1:
            return silhouette_score(self.data, self.clusters)
        else:
            return -1 

    def completeness(self):
        return completeness_score(self.target, self.clusters)

    def map_clusters_to_labels(self):
        cluster_to_label = {}
        for cluster in range(self.k):
            cluster_indices = np.where(self.clusters == cluster)[0]
            true_labels = self.target[cluster_indices]
            if len(true_labels) > 0:
                most_common_label = np.bincount(true_labels).argmax()
                cluster_to_label[cluster] = most_common_label
        mapped_labels = np.array([cluster_to_label[cluster] for cluster in self.clusters])
        self.cluster_to_label = cluster_to_label
        return mapped_labels, cluster_to_label

    def accuracy(self):
        _, cluster_to_label = self.map_clusters_to_labels()
        cluster_accuracies = {}
        for cluster, label in cluster_to_label.items():
            cluster_indices = np.where(self.clusters == cluster)[0]
            true_labels = self.target[cluster_indices]
            accuracy = np.mean(true_labels == label)
            cluster_accuracies[cluster] = accuracy
        mean_accuracy = np.mean(list(cluster_accuracies.values()))
        accuracies_mapped = {cluster: (label, acc) for cluster, label, acc in zip(cluster_to_label.keys(), cluster_to_label.values(), cluster_accuracies.values())}
        return accuracies_mapped, mean_accuracy

    def summary(self):
        cluster_accuracies, acc = self.accuracy()
        cluster_to_label = self.cluster_to_label
        silhouette_score = self.silhouette()
        completeness = self.completeness()
        s = self.__str__() + f"\nMean Accuracy: {acc * 100:.2f}%\nCluster to Label Mapping: {cluster_to_label}\nSilhouette Score: {silhouette_score}\nCompleteness Score: {completeness}\nCluster Accuracies: {cluster_accuracies}\n"
        print(s)
    
    def run(self):
        self.fit()
        self.map_clusters_to_labels()

    @classmethod
    def find_best_k(cls, k_values, tol=1e-4, max_iter=1000, random_state=42):
        results = []
        for k in k_values:
            kmeans = cls(k=k, tol=tol, max_iter=max_iter, random_state=random_state)
            kmeans.run()
            map_clusters = kmeans.cluster_to_label
            clusters_not_found = [i for i in range(10) if i not in map_clusters.values()]
            _, mean_accuracy = kmeans.accuracy()
            silhouette = kmeans.silhouette()
            completeness = kmeans.completeness()
            if len(clusters_not_found) == 0:
                results.append((k, mean_accuracy, silhouette, completeness))
        return results
