from dataset import digits_dataset
from sklearn.metrics import confusion_matrix
import numpy as np
from sklearn.metrics import pairwise_distances_argmin, silhouette_score, completeness_score

class Kmeans:
    def __init__(self, k=10, tol=1e-4, max_iter=1000, random_state=42):
        self.data = digits_dataset.training_data
        self.test = digits_dataset.test_data
        self.target = digits_dataset.training_target
        self.test_target = digits_dataset.test_target
        self.k = k
        self.max_iter = max_iter
        self.random_state = random_state
        self.tol = tol
        self.centroids = None
        self.clusters = None
        self.clusters_map = None

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

    def predicted_number(self):
        predicted = [self.clusters_map[x] for x in self.clusters]
        self.predicted_table = ["OK" if predicted[x] == self.target[x] else f"Predicted {predicted[x]} but had {self.target[x]} " for x in range(len(self.target))]
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
        clusters_map = {}
        for cluster in range(self.k):
            cluster_indices = np.where(self.clusters == cluster)[0]
            true_labels = self.target[cluster_indices]
            if len(true_labels) > 0:
                most_common_label = np.bincount(true_labels).argmax()
                clusters_map[cluster] = most_common_label
        mapped_labels = np.array([clusters_map[cluster] for cluster in self.clusters])
        self.clusters_map = clusters_map
        return mapped_labels, clusters_map

    def accuracy(self):
        _, clusters_map = self.map_clusters_to_labels()
        cluster_accuracies = {}
        for cluster, label in clusters_map.items():
            cluster_indices = np.where(self.clusters == cluster)[0]
            true_labels = self.target[cluster_indices]
            accuracy = np.mean(true_labels == label)
            cluster_accuracies[cluster] = accuracy
        mean_accuracy = np.mean(list(cluster_accuracies.values()))
        accuracies_mapped = {cluster: (label, acc) for cluster, label, acc in zip(clusters_map.keys(), clusters_map.values(), cluster_accuracies.values())}
        return accuracies_mapped, mean_accuracy

    def summary(self):
        cluster_accuracies, acc = self.accuracy()
        clusters_map = self.clusters_map
        silhouette_score = self.silhouette()
        completeness = self.completeness()
        s = self.__str__() + f"\nMean Accuracy: {acc * 100:.2f}%\nCluster to Label Mapping: {clusters_map}\nSilhouette Score: {silhouette_score}\nCompleteness Score: {completeness}\nCluster Accuracies: {cluster_accuracies}\n"
        print(s)
    
    def run(self):
        self.fit()
        self.map_clusters_to_labels()
        self.predicted_number()

    @classmethod
    def find_best_k(cls, k_values, tol=1e-4, max_iter=1000, random_state=42):
        results = []
        for k in k_values:
            kmeans = cls(k=k, tol=tol, max_iter=max_iter, random_state=random_state)
            kmeans.run()
            map_clusters = kmeans.clusters_map
            clusters_not_found = [i for i in range(10) if i not in map_clusters.values()]
            _, mean_accuracy = kmeans.accuracy()
            silhouette = kmeans.silhouette()
            completeness = kmeans.completeness()
            if len(clusters_not_found) == 0:
                results.append((k, mean_accuracy, silhouette, completeness))
        return results

    def confusion_matrix(self):
        real_vals = [self.clusters_map[x] for x in self.clusters]
        return confusion_matrix(self.target, real_vals)