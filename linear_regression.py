"""
ML Algorithm that uses linear regression to predict survivors on the ]
Titanic based on information of the person
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from tensorflow import feature_column as fc

import tensorflow as tf


# Pull Training Data and Load into Pandas DataFrame
# survived - 1 = yes, 0 = no
# sex - Sex of Passenger
# age - Age of Passenger
# n_siblings_spouses - Number of Siblings and Spouses on Board
# parch - Number of Parents/Children Aboard
# fare - Passenger Fare in British Pounds
# class - Passenger Class
# deck - What Deck the Passenger was on
# embark_town - Port of Embarkation
# alone - y = yes, n = no
dftrain = pd.read_csv('https://storage.googleapis.com/tf-datasets/titanic/train.csv')
dfeval = pd.read_csv('https://storage.googleapis.com/tf-datasets/titanic/eval.csv')

# Seperate and Store 'Answer' Data
y_train = dftrain.pop('survived')
y_eval = dfeval.pop('survived')

