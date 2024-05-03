

from dataset import digits_dataset
import matplotlib.pyplot as plt
import numpy as np
from pprint import pprint
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