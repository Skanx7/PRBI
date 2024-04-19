import pandas as pd
import numpy as np

PenguinData = pd.read_csv('datasets/penguins_classification.csv')

nb_species = PenguinData["Species"].value_counts()
def apriori_probabilities():
    apriori_probabilities = {}
    for species in nb_species.index:
        apriori_probabilities[species] = nb_species[species] / len(PenguinData)
    return apriori_probabilities
print(nb_species)
print(apriori_probabilities())