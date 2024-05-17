from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split
class Dataset:
    _feature_names = ["zero", "un", "deux", "trois", "quatre", "cinq", "six", "sept", "huit", "neuf"]
    def __init__(self):
        self.name = "Optical recognition of handwritten digits dataset"
        self.dataset = load_digits()
        self.description = self.dataset["DESCR"][123:1165]
        self.images = self.dataset["images"]
        self.data = self.dataset["data"]
        self.training_data, self.test_data = train_test_split(self.data, test_size=0.2, shuffle=False)
        self.target = self.dataset["target"]
        self.training_target, self.test_target = train_test_split(self.target, test_size=0.2, shuffle=False)
    def __str__(self):
        return f"""Dataset: {self.name}
Description: {self.description}
Images: {self.images.shape}
Data: {self.data.shape}
Targets: {len(self.target)}"""


digits_dataset = Dataset()
