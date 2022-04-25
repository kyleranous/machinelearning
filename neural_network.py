import tensorflow as tf
from tensorflow import keras

import numpy as np
import matplotlib.pyplot as plt

fashion_mnist = keras.datasets.fashion_mnist # load dataset

(train_images, train_labels), (test_images, test_labels) = fashion_mnist.load_data()

class_names = ['T-Shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat', 'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot']

# Preprocess the images
train_images = train_images / 255.0
test_images = test_images / 255.0

# Define the Model
model = keras.Sequential([
    keras.layers.Flatten(input_shape=(28, 28)),  # input layer (1)
    keras.layers.Dense(128, activation='relu'),  # hidden layer (2)
    keras.layers.Dense(10, activation='softmax') # output layer (3)
])

# Compile the Model
model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

# Train the Model
model.fit(train_images, train_labels, epochs=5)

# Evaluate the model
#test_loss, test_acc = model.evaluate(test_images, test_labels, verbose=1)

#print('Test accuracy: ', test_acc)

predictions = model.predict(test_images)

print(class_names[np.argmax(predictions[7000])])

plt.figure()
plt.imshow(test_images[7000])
plt.colorbar()
plt.grid(False)
plt.show()