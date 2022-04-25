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

# Process Catagorical Data
CATEGORICAL_COLUMNS = ['sex', 'n_siblings_spouses', 'parch', 'class', 'deck', 'embark_town', 'alone']

NUMERIC_COLUMNS = ['age', 'fare']

feature_columns = []
for feature_name in CATEGORICAL_COLUMNS:
    vocabulary = dftrain[feature_name].unique() # Pulls a list of each unique value in a column
    feature_columns.append(tf.feature_column.categorical_column_with_vocabulary_list(feature_name, vocabulary))

for feature_name in NUMERIC_COLUMNS:
    feature_columns.append(tf.feature_column.numeric_column(feature_name, dtype=tf.float32))