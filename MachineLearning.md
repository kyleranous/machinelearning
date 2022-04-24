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


Linear Regression is one of the most basic forms of machine learning and is used to predict numeric values.

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

ML algorithms require data

## Classification

## Clustering

## Hidden Markov Models

## References

- [freeCodeCamp](https://www.freecodecamp.org/learn/machine-learning-with-python/)