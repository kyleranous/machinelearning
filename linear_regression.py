import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from tensorflow import feature_column as fc

import tensorflow as tf


# Pull Training Data and Load into Pandas DataFrame
dftrain = pd.read_csv('https://storage.googleapis.com/tf-datasets/titanic/train.csv')
dfeval = pd.read_csv('https://storage.googleapis.com/tf-datasets/titanic/eval.csv')

# Seperate and Store 'Answer' Data
y_train = dftrain.pop('survived')
y_eval = dftrain.pop('survived')

