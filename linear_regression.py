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


# Create Input function for processing data to be fed to model
def make_input_fn(data_df, label_df, num_epochs=10, shuffle=True, batch_size=32):
    def input_function(): # Function returned by input function
        ds = tf.data.Dataset.from_tensor_slices((dict(data_df), label_df)) # Create tf.data.Dataset object from the pandas data and labels
        if shuffle:
            ds = ds.shuffle(1000) # Randomize the order of the data
        ds = ds.batch(batch_size).repeat(num_epochs) # Split the dataset into batches of size batch_size and repeat process for the number of epochs
        return ds
    return input_function

# Setup the Training and Evaluation input functions
train_input_fn = make_input_fn(dftrain, y_train)
eval_input_fn = make_input_fn(dfeval, y_eval, num_epochs=1, shuffle=False)

# Create the ML Model
linear_est = tf.estimator.LinearClassifier(feature_columns=feature_columns)

# Train the model with training data
linear_est.train(train_input_fn) # Train the model with training data passed from the training input function

result = linear_est.evaluate(eval_input_fn) # Evaluate the model with the eval dataset passed from the training input function, result will hold the model metrics/stats

print(result['accuracy'])