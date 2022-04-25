# Machine Learning

*These are my personal notes on machine learning. These notes are collected from multiple trainings and tutorials as noted in the [References](#references) section*

Machine Learning is the use and development of computer systems and models that are able to learn and adapt without following explicit instructions, by using algorithms and statistical models to analyze and infer from patterns of data. 

## TOC

 - [Tech Stack](#tech-stack)
 - [TensorFlow](#tensorflow)
 - [Machine Learning Algorithms](#machine-learning-algorithms)
    - [Linear Regression](#linear-regression)
    - [Classification]()
    - [Clustering]()
    - [Hidden Markov Models]()
 - [References](#references)


## Tech Stack
Packages Used:
 - TensorFlow `pip install tensorflow`
 - pandas `pip install pandas`
 - matplotlib `pip install matplotlib`
 - numpy `pip install numpy`
 - TensorFlow-Probability `pip install tensorflow-probability`


## TensorFlow
TensorFlow is a FOSS library for Machine Learning and AI with a focus on training and inference of Deep Neural Networks(DNN). It was developed by the Google Brain Team. 

https://www.tensorflow.org/

### Tensors

A tensor is a generalization of vectors and matrices to potentially higher dimensions. Internally, TensorFlow represents tensors as n-dimensional arrays of base datatypes."

**Types of Tensors**
 - Variable
 - Constant
 - Placeholder
 - SparseTensor

 With the exception of `Variable` all of these tensors are immutable.

#### Creating Tensors

```python
string = tf.Variable("this is a string", tf.string)
number = tf.Variable(324, tf.int16)
floating = tf.Variable(3.567, tf.float64)
```

#### Rank/Degree of Tensors

Ranks or Degree is the number of embedded arrays in a tensor. The above example are *Rank 0* as it is not an array (list) and can only hold a single value.

```python
rank1_tensor = tf.Variable(['This', 'is', 'a', 'rank', '1', 'tensor'], tf.string)
rank2_tensor = tf.Variable([['This', 'is', 'a'], ['rank', '2', 'tensor']], tf.string)
```

Determining the Rank of a Tensor:
```python
>>> tf.rank(rank2_tensor)
<tf.Tensor: shape=(), dtype=int32, numpy=2>
```

#### Tensor Shape

The shape of a tensor is the amount of elements that exist in each dimension.

```python
>>> rank2_tensor.shape
TensorShape([2, 3])
```

#### Changing Shape

The number of elements of a tensor is the product of the sizes of all its shapes. There are often many shapes that have the same number of elements making it convenient to be able to change the shape of a tensor.

```python
>>> tensor1 = tf.ones([1,2,3]) # Creates a tensor with 1 list, consisting of 2 lists with 3 elements each - 6 elements total
>>> print(tensor1)
tf.Tensor(
[[[1. 1. 1.]
  [1. 1. 1.]]], shape=(1, 2, 3), dtype=float32)

>>> tensor2 = tf.reshape(tensor1, [2,3,1]) # Reshape Tensor to 2 lists, consisting of 3 lists each, each containing 1 element - 6 elements total
>>> print(tensor2)
tf.Tensor(
[[[1.]
  [1.]
  [1.]]

 [[1.]
  [1.]
  [1.]]], shape=(2, 3, 1), dtype=float32)

>>> tensor3 = tf.reshape(tensor2, [3, -1]) # A value of -1 tells tensor flow to calculate that value to ensure the tensor is reshaped accordingly
>>> print(tensor3)
tf.Tensor(
[[1. 1.]
 [1. 1.]
 [1. 1.]], shape=(3, 2), dtype=float32)
```

## Machine Learning Algorithms

 - Linear Regression
 - Classification
 - Clustering
 - Hidden Markov Models

### Linear Regression

[Example Script](linear_regression.py)


Linear Regression is one of the most basic machine learning algorithms and is used to predict numeric values.

Linear Regression Function: `y = mx + b`

Linear Regression calculates the line of best fit for a dataset. 

Given the following plot:

![Linear Regression Sample Plot](/images/linear_regression/linear_regression_example_plot.png)

with points at (1,1), (2,4), (2.5,7), (3,9), and (4,15)

The calculated line of best fit would be:

![Line of Best Fit](/images/linear_regression/linear_regression_LOBF.png)

The line of best fit has the equation: `y = 4.7x - 4.55`

*Graphs Generated with matplotlib*
<details>
    <summary>View Code</summary>

```python
import matplotlib.pyplot as plt
import numpy as np

# Sample Dataset
x = [1, 2, 2.5, 3, 4]
y = [1, 4, 7, 9, 15]

# Plot Sample Dataset
plt.plot(x,y, 'ro')
plt.axis([0, 6, 0, 20])

# Calculate and Plot Line of Best Fit
plt.plot(np.unique(x), np.poly1d(np.polyfit(x, y, 1))(np.unique(x)))

plt.show()

# Finding Values of m and b =? [ m b ]
f = np.polyfit(x, y, 1)
print(f)
```
</details>

#### Data

ML algorithms require data to train and evaluate models. Data can come in many different forms. 

Machine learning algorithms require seperate training and evaluation data to ensure the algorithm hasn't just memorized the training data.

The dataset used for [linear_regression.py](/linear_regression.py) is a sample dataset of titanic survivors from TensorFlow. 
<br>

Training Data: https://storage.googleapis.com/tf-datasets/titanic/train.csv
<br>
Evaluation Data: https://storage.googleapis.com/tf-datasets/titanic/eval.csv

**Training Data Graphs**

![Titanic Passengers by age](/images/linear_regression/titanic_passengers_by_age.png)

<details>
    <summary>Graph Code</summary>

```python
dftrain.age.plot.hist(bins=20)
plt.title('Passengers by age')
plt.xlabel('Age')
plt.show()
```
</details>


![Titanic Passengers by sex](/images/linear_regression/titanic_passengers_by_sex.png)


<details>
    <summary>Graph Code</summary>

```python
dftrain.sex.value_counts().plot(kind='barh')
plt.title('Passengers by Sex')
plt.show()
```
</details>


![Titanic Passengers by class](/images/linear_regression/titanic_passenger_by_class.png)

<details>
    <summary>Graph Code</summary>

```python
dftrain['class'].value_counts().plot(kind='barh')
plt.title('Passengers by Class')
plt.show()
```
</details>


![Titanic Survivors By Sex](/images/linear_regression/titanic_survivors_by_sex.png)

<details>
    <summary>Graph Code</summary>

```python
pd.concat([dftrain, y_train], axis=1).groupby('sex').survived.mean().plot(kind='barh').set_xlabel('% survived')
plt.title('Titanic Survivors by Sex')
plt.show()
```
</details>

**Catagorical vs Feature Data**
Catagorical data is data that is not numeric. It will need to be converted to numeric data before it can be used

Processing Catagorical and Feature Columns:
```python
CATEGORICAL_COLUMNS = ['sex', 'n_siblings_spouses', 'parch', 'class', 'deck', 'embark_town', 'alone']

NUMERIC_COLUMNS = ['age', 'fare']

feature_columns = []
for feature_name in CATEGORICAL_COLUMNS:
    vocabulary = dftrain[feature_name].unique() # Pulls a list of each unique value in a column
    feature_columns.append(tf.feature_column.categorical_column_with_vocabulary_list(feature_name, vocabulary))

for feature_name in NUMERIC_COLUMNS:
    feature_columns.append(tf.feature_column.numeric_column(feature_name, dtype=tf.float32))
```

#### Training Process

In linear_regression.py training data is streamed in *batchs* of 32 datasets for a number of *epochs*. An *epoch* is the number of times the model will see the complete dataset. 

Loading data in batches is usefull for very large datasets that can't all be loaded into memory. 

**Over Fitting**<br>
It is possible to feed the data to many times to a model. The model then *memorizes* the dataset and is really good at predicting the training data, but fails with the evaluation data. Preventing this is done by starting with a low number of epochs and slowly increasing to tune the best results.

![accuracy vs epoch data](./images/linear_regression/accuracy_vs_epochs.png)


Converting data into batchs and feeding it to the training model is done through an *input function*

This TensorFlow model requires data be passed as a td.data.Dataset object. The input function converts the pandas dataframe object into a td.data.Dataset object. 

```python
def make_input_fn(data_df, label_df, num_epochs=10, shuffle=True, batch_size=32):
    def input_function(): # Function returned by input function
        ds = tf.data.Dataset.from_tensor_slices((dict(data_df), label_df)) # Create tf.data.Dataset object from the pandas data and labels
        if shuffle:
            ds = ds.shuffle(1000) # Randomize the order of the data
        ds = ds.batch(batch_size).repeat(num_epochs) # Split the dataset into batches of size batch_size and repeat process for the number of epochs
        return ds
    return input_function
```

make_input_fn passes an inner function that can then be called by the model to process the initial data. (Think Lambda Functions)

#### Creating the Model

TensorFlow uses *estimators* to create the ML models.

```python
linear_est = tf.estimator.LinearClassifier(feature_columns=feature_columns)
```
The above code creates an estimator that will use a LinearClassifier from the feature_columns that are passed to it.

#### Training and Evaluating the Model

Passing the estimator model the training data in *train* mode will train the data
```python
linear_est.train(train_input_fn)
```

Passing the estimator model the eval data in *evaluate* mode will evaluate the data and return metrics / stats for the model
```python
result = linear_est.evaluate(eval_input_fn)
```
Evaluating a model returns a dictionary.

##### Predicting results
models have a predict method that will return..... a prediction. in the case of linear_regression.py the model will predect weather the passenger survived or died.

```python
result = list(linear_est.predict(eval_input_fn)) # Convert to list to parse the results
```

the prediction can be accessed through the `probabilities` dictionary object. `probabilities` returns an array with the calculated propabilities of the results. In the case of linear_regression.py the array is [died, survived] so returning `result[0]['probabilities'][1]` would return the probability that the passenger at index 0 survived. 
### Classification
Classification is used to seperate datapoints into classes of different labels. 

Example uses the Iris Flower Dataset from TensorFlow
 - Training Data: https://storage.googleapis.com/download.tensorflow.org/data/iris_training.csv
 - Evaluation Data: https://storage.googleapis.com/download.tensorflow.org/data/iris_test.csv

#### Data

The data in this example seperates flowers into 3 different classes of species:
 - Setosa
 - Versicolor
 - Virginica

The data provided for each flower is:
 - Sepal Length
 - Sepal Width
 - Petal Length
 - Petal Width

Defining Column Names:
```python
CSV_COLUMN_NAMES = ['SepalLength', 'SepalWidth', 'PetalLength', 'PetalWidth', 'Species']
SPECIES = ['Serosa', 'Versicolor', 'Virginica']
```

Output of test.head()
```bash
   SepalLength  SepalWidth  PetalLength  PetalWidth  Species
0          6.4         2.8          5.6         2.2        2
1          5.0         2.3          3.3         1.0        1
2          4.9         2.5          4.5         1.7        2
3          4.9         3.1          1.5         0.1        0
4          5.7         3.8          1.7         0.3        0
```

Remove the results (Species) column from the training and evaluation data

```python
train_y = train.pop('Species')
test_y = test.pop('Species')
```

#### Input Function

```python
def input_fn(features, labels, training=True, batch_size=256):
    # Convert the inputs into a Dataset
    dataset = td.data.Dataset.from_tensor_slices((dict(features), labels))

    # Shuffle and repeat if in training mode
    if training:
        dataset = dataset.shuffle(1000).repeat()

    return dataset.batch(batch_size)
```
This input function is simplier then the input function used in the linear regression model. This function is only returning a dataset that has been sliced into batches. As such, when we evaluate, we will have to evaluate a lambda function that gets this method passed to it and sets epochs.

#### Feature Columns
Since the data has already been encoded (No Catigorical data) we don't need to worry about a vocab map.

```python
# Feature columns describe how to use the input
my_feature_columns = []
for key in train.keys():
    my_feature_columns.append(tf.feature_column.numeric_column(key=key))

```
This function loops through the column headers in the train dataset (`train.keys()`) and appends them to a list of feature columns that we will pass to the estimator.

#### Building the Model

TensorFlow reccomends using a DNNClassifier (Deep Neural Network) model for classification

```python
# Build a DNN with 2 hidden layers with 30 and 10 hidden nodes each.
classifier = tf.estimator.DNNClassifier(feature_columns=my_feature_columns,
    # Two hidden laters of 30 and 10 nodes respectively.
    hidden_units=[30, 10],
    # The model must choose between 3 classes.
    n_classes = 3)
```
*tf.estimator is depreciated in TF 2.0. It is reccomended to use kera's instead*

*What are hidden layers?*<br>
*What are hidden nodes?*

#### Training the Model

```python
classifier.train(
    input_fn=lambda: input_fn(train, train_y, training=True),
    steps=5000)
```

We pass a lambda function to classifier.train, with the input_fn() that was defined above. We are passing a lambda function because input_fn() returns the processed dataset, but not the function to do it.

`steps=5000` is an alternative to calling out epochs. If we train a dataset with 2000 entries, with a batch size of 10, then an epoch consists of 2000 entries / (10 entries / step) = 200 steps. so calling out steps=5000 runs 5000 batches of data, and may not be a hole number of epochs.

#### Evaluate the Model

```python
eval_result = classifier.evaluate(input_fn=lambda: input_fn(test, test_y, training=False))
print(f'\nTest set accuracy: {format(eval_result["accuracy"], "0.3f")}\n')
```

#### Predictions based on user input

```python
# Create a new Input Function to process User provided Data
def user_input_fn(features, batch_size=256):
    # Convert user inputs into a Dataset without labels
    return tf.data.Dataset.from_tensor_slices(dict(features)).batch(batch_size)


features = ['SepalLength', 'SepalWidth', 'PetalLength', 'PetalWidth']
predict = {}
print("Please type numeric values as prompted.")
for feature in features:
    valid = True
    while valid:
        val = input(f'{feature}: ')
        if not val.isdigit(): valid = False

    predict[feature] = [float(val)]

predictions = classifier.predict(input_fn=lambda: user_input_fn(predict))
for pred_dict in predictions:
    class_id = pred_dict['class_ids'][0]
    probability = pred_dict['probabilities'][class_id]

    print(f'Prediction is "{SPECIES[class_id]}" ({format(100 * probability, ".3f")})')
```

### Clustering
Clustering is a technique that involves the grouping of data points. It is used when you have lots of datapoints for features, but no information on labels. It works by grouping datapoints that have similar properties/features. New data is plotted checked against the model and grouped with the other data.

https://towardsdatascience.com/the-5-clustering-algorithms-data-scientists-need-to-know-a36d136ef68

K-Means clustering is the most well known clustering algorithm

**K-Means Clustering**
1. Select the number of classes/groups to use and randomly place their centroids
2. Assign all datapoints to the centroids by distance, the closest centroid to a datapoint is the class of that datapoint
3. Average the datapoints assigned to each centroid to recalculate the centroid position
4. Reassigne each datapoint a centroid based on the new central location
5. Repeat Steps 3-4 until no points change which centroid they belong to

### Hidden Markov Models

Hidden Markov Model is a finite set of statesm each of which is associated with a (generally multidimensional) probability distribution. Transistions among the states are governed by a set of prabilities called transition probabilities

A hidden markov movel works with probabilities to predict future events or states.

#### Data

**States**: In each markov model we have a finite set of states, these states are "hidden" within the model which means we do not directly observe them.

**Observations**: Each state has a particular outcome or observation associated with it based on a probability distribution. ex: 
> On a hot day Time has an 80% chance of being happy and a 20% chance of being sad

**Transitions**: Each state will have a probability defining the likelyhood of transitioning to a different state.
> A cold day has a 30% chance of being followed by a hot day and a 70% chance of being followed by another cold day.

To create a Hidden Markov Model we need:
 - States
 - Observation Distribution
 - Transition Distribution

 #### Weather Model

 From TensorFlow documentation
 https://www.tensorflow.org/probability/api_docs/python/tfp/distributions/HiddenMarkovModel

 Model a simple weather system and try to predict the temperature on each day given the following information:

 1. Cold days are encoded by a 0 and hot days are encoded by a 1
 1. The first day in our sequence has an 80% chance of being cold
 1. A cold day has a 30% chance of being followed by a hot day
 1. A hot day has a 20% chance of being followed by a cold day
 1. On each day yhe temperature is normally distributed with mean and standard deviation 0 and 5 on a cold day and mean and standard deviation 15 and 10 on a hot day

On a hot day the average temperature is 15 and ranges from 5 to 25

Creating the distribution variables:
```python
tfd = tfp.distributions # Shortcut for use lateron
initial_distribution = tfd.Categorical(probs=[0.8, 0.2])
transition_distribution = tfd.Categorical(probs=[[0.7, 0.3],
                                                [0.2, 0.8]])
observation_distribution = tfd.Normal(loc=[0., 15.], scale=[5., 10.])
# the loc argument represents the mean and the scale is the standard deviation
```

Creating the model:
```python
model = tfd.HiddenMarkovModel(
    initial_distribution=initial_distribution,
    transition_distribution=transition_distribution,
    observation_distribution=observation_distribution,
    num_steps=7)
```

Viewing the Model Results:
```python
print(model.numpy())
```
Hidden Markov Models don't need to be trained as they are operating purly off probabilities. As long as the probabilities havn't changed, the calculations will always be the same.

## Neural Networks

## Keras

## References

- [freeCodeCamp](https://www.freecodecamp.org/learn/machine-learning-with-python/)
- [TensorFlow Tutorials](https://www.tensorflow.org/tutorials)