from dataset import digits_dataset
import matplotlib.pyplot as plt
import numpy as np
from pprint import pprint
from sklearn.decomposition import PCA      # Juste pour représenter les graphes sous format 2D
from sklearn.manifold import TSNE
from sklearn.metrics import silhouette_score
from kmeans import Kmeans  

class Visualizer:
    """
    A class for visualizing datasets.

    Attributes:
        _instance (Visualizer): The singleton instance of the Visualizer class.
        dataset: The dataset to be visualized.
        description (str): The description of the visualizer.

    Methods:
        __init__(self, dataset=None): Initializes a Visualizer object.
        __str__(self): Returns a string representation of the Visualizer object.
        get_instance(cls): Returns the singleton instance of the Visualizer class.
        get_indices_index(self, index_to_search, amount): Returns the indices of a specific target index in the dataset.
        show_indexes(cls, index_list, colored=False): Displays images corresponding to the given index list.
        show_index_multiple(cls, index=3, amount=10, colored=False): Displays images for multiple indices of a specific target index.
        show(cls, amount=10, colored=False): Displays images for each target class in the dataset.
    """
    _instance = None

    def __init__(self, dataset=None):
        """
        Initializes a Visualizer object.

        Args:
            dataset: The dataset to be visualized. If None, the default digits_dataset will be used.
        """
        self.description = ""
        if dataset is None:
            self.dataset = digits_dataset
        else:
            self.dataset = dataset

    def __str__(self):
        """
        Returns a string representation of the Visualizer object.

        Returns:
            str: A string representation of the Visualizer object.
        """
        return f"""Visualizer for {self.dataset.name}
Instance: {self.__repr__()}"""

    @classmethod
    def get_instance(cls):
        """
        Returns the singleton instance of the Visualizer class.

        Returns:
            Visualizer: The singleton instance of the Visualizer class.
        """
        if cls._instance is None:
            cls._instance = cls()
        return cls._instance

    def get_indices_index(self, index_to_search, amount):
        """
        Returns the indices of a specific target index in the dataset.

        Args:
            index_to_search: The target index to search for.
            amount: The maximum number of indices to return.

        Returns:
            list: The indices of the target index in the dataset.
        """
        indices = []
        i = 0
        while len(indices) < amount and i < len(self.dataset.target):
            if self.dataset.target[i] == index_to_search:
                indices.append(i)
            i += 1
        return indices

    @classmethod
    def show_number_matrix(cls, index, threshold=8):
        """
        Displays the matrix representation of the image corresponding to the given index.

        Args:
            index: The index of the image to display.
        """
        instance = cls.get_instance()
        num = instance.dataset.images[index]
        mat_num = np.where(num > threshold, "*", " ")
        pprint(mat_num)

    @classmethod
    def show_indexes(cls, index_list, colored=False):
        """
        Displays images corresponding to the given index list.

        Args:
            index_list: The list of indices to display.
            colored (bool): Whether to display the images in color or grayscale. Default is False.
        """
        instance = cls.get_instance()
        cmap = None if colored else 'gray'
        amount = len(index_list)
        columns = int(np.ceil(np.sqrt(amount)))
        rows = int(np.ceil(amount / columns))
        fig, axes = plt.subplots(rows, columns, figsize=(5 * columns, 5 * rows))
        if rows * columns == 1:
            axes = [axes]
        else:
            axes = axes.flatten()
        for i, index in enumerate(index_list):
            if i >= amount:
                break
            axes[i].imshow(instance.dataset.images[index], cmap=cmap)
            axes[i].set_title(f'Index: {index} - Target: {instance.dataset.target[index]}')
        # Set the overall figure title
        fig.suptitle('Visualizer')
        fig.canvas.manager.set_window_title(f'Visualizer - For the index list {index_list}')

        plt.tight_layout(pad=3.0)  # Adjust layout to prevent overlap
        plt.show()

    @classmethod
    def show_index_multiple(cls, index=3, amount=10, colored=False):
        """
        Displays images for multiple indices of a specific target index.

        Args:
            index: The target index to display images for. Default is 3.
            amount: The number of images to display. Default is 10.
            colored (bool): Whether to display the images in color or grayscale. Default is False.
        """
        instance = cls.get_instance()
        indices = instance.get_indices_index(index, amount)
        cls.show_indexes(indices, colored)

    @classmethod
    def show(cls, amount=10, colored=False):
        """
        Displays images for each target class in the dataset.

        Args:
            amount: The number of images to display for each target class. Default is 10.
            colored (bool): Whether to display the images in color or grayscale. Default is False.
        """
        instance = cls.get_instance()
        indices = {}
        cmap = None if colored else 'gray'
        classes = instance.dataset._feature_names
        for i in range(10):
            indices[i] = instance.get_indices_index(i, amount)
        plt.figure(figsize=(10, 10))
        plt.suptitle('Visualizer')
        for i in range(len(classes)):
            for j in range(amount):
                plt.subplot(len(classes), amount, amount * i + j + 1)
                plt.xticks([])
                plt.yticks([])
                plt.grid(False)
                plt.imshow(instance.dataset.images[indices[i][j]], cmap=cmap)
                if j == 0:
                    plt.ylabel(instance.dataset._feature_names[i])
        plt.tight_layout()
        plt.show()

    @classmethod
    def show_clusters(cls, clusters, map_clusters, amount=10, flag_correct=False, flag_incorrect=False):
        """
        Displays images for each cluster in the dataset.

        Args:
            clusters: The array of cluster assignments for the dataset.
            map_clusters: The mapping from cluster to label.
            amount: The number of images to display for each cluster. Default is 10.
            flag_correct: Boolean to flag correctly predicted images. Default is False.
            flag_incorrect: Boolean to flag incorrectly predicted images. Default is False.
        """
        instance = cls.get_instance()
        classes = instance.dataset._feature_names
        cmap = 'gray'
        plt.figure(figsize=(10, 10))
        clusters_not_found = [i for i in range(10) if i not in map_clusters.values()]
        titre = 'Clusters Visualizer'+ f" (Missing good clusters for : {clusters_not_found})" if len(clusters_not_found) > 0 else ''
        plt.suptitle(titre)

        for cluster in range(np.max(clusters) + 1):
            indices = np.where(clusters == cluster)[0][:amount]
            for i, index in enumerate(indices):
                plt.subplot(np.max(clusters) + 1, amount, cluster * amount + i + 1)
                plt.xticks([])
                plt.yticks([])
                plt.grid(False)
                plt.imshow(instance.dataset.images[index], cmap=cmap)

                # Get the true label and predicted label
                true_label = instance.dataset.target[index]
                predicted_label = map_clusters[cluster]

                # Set the title with the true label
                plt.xlabel(f'{classes[true_label]}')

                # Highlight the border
                if flag_correct and true_label == predicted_label:
                    plt.gca().spines['top'].set_color('green')
                    plt.gca().spines['top'].set_linewidth(3)
                    plt.gca().spines['bottom'].set_color('green')
                    plt.gca().spines['bottom'].set_linewidth(3)
                    plt.gca().spines['left'].set_color('green')
                    plt.gca().spines['left'].set_linewidth(3)
                    plt.gca().spines['right'].set_color('green')
                    plt.gca().spines['right'].set_linewidth(3)
                elif flag_incorrect and true_label != predicted_label:
                    plt.gca().spines['top'].set_color('red')
                    plt.gca().spines['top'].set_linewidth(3)
                    plt.gca().spines['bottom'].set_color('red')
                    plt.gca().spines['bottom'].set_linewidth(3)
                    plt.gca().spines['left'].set_color('red')
                    plt.gca().spines['left'].set_linewidth(3)
                    plt.gca().spines['right'].set_color('red')
                    plt.gca().spines['right'].set_linewidth(3)

                if i == 0:
                    cluster_label = map_clusters[cluster]
                    plt.ylabel(f'{classes[cluster_label]}')
        plt.tight_layout()
        plt.show()

    @classmethod
    def plot_clusters(cls, kmeans):
        instance = cls.get_instance()
        tsne = TSNE(n_components=2, random_state=kmeans.random_state)
        reduced_data = tsne.fit_transform(kmeans.data)

        plt.figure(figsize=(10, 10))
        for cluster in range(kmeans.k):
            indices = np.where(kmeans.clusters == cluster)
            plt.scatter(reduced_data[indices, 0], reduced_data[indices, 1], label=f'Cl. {cluster} => ({instance.dataset._feature_names[kmeans.clusters_map[cluster]]})')
        
        # Calculate silhouette score in the t-SNE space
        silhouette = silhouette_score(reduced_data, kmeans.clusters)

        plt.legend()
        plt.title(f'Cluster Visualization using t-SNE\nSilhouette Score: {silhouette:.2f}')
        plt.xlabel('tSNE-C1')
        plt.ylabel('tSNE-C2')
        plt.show()

    @staticmethod
    def plot_best_k_metrics(results):
        """
        Plots mean accuracy, silhouette score, and completeness score for different values of k.

        Args:
            results: List of tuples containing (k, mean_accuracy, silhouette, completeness)
        """
        # Extract the metrics from the results
        ks = [result[0] for result in results]
        mean_accuracies = [result[1] for result in results]
        silhouettes = [result[2] for result in results]
        completenesses = [result[3] for result in results]

        # Plotting the metrics
        plt.figure(figsize=(14, 8))

        plt.subplot(1, 3, 1)
        plt.plot(ks, mean_accuracies, marker='o')
        plt.title('Mean Accuracy vs. K')
        plt.xlabel('K')
        plt.ylabel('Mean Accuracy')

        plt.subplot(1, 3, 2)
        plt.plot(ks, silhouettes, marker='o')
        plt.title('Silhouette Score vs. K')
        plt.xlabel('K')
        plt.ylabel('Silhouette Score')

        plt.subplot(1, 3, 3)
        plt.plot(ks, completenesses, marker='o')
        plt.title('Completeness Score vs. K')
        plt.xlabel('K')
        plt.ylabel('Completeness Score')

        plt.tight_layout()
        plt.show()