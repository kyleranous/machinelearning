import tensorflow as tf
import pandas as pd


CSV_COLUMN_NAMES = ['SepalLength', 'SepalWidth', 'PetalLength', 'PetalWidth', 'Species']
SPECIES = ['Serosa', 'Versicolor', 'Virginica']

train_path = tf.keras.utils.get_file('iris_training.csv', 'https://storage.googleapis.com/download.tensorflow.org/data/iris_training.csv')

test_path = tf.keras.utils.get_file('iris_test.csv', 'https://storage.googleapis.com/download.tensorflow.org/data/iris_test.csv')

train = pd.read_csv(train_path, names=CSV_COLUMN_NAMES, header=0)
test = pd.read_csv(test_path, names=CSV_COLUMN_NAMES, header=0)

train_y = train.pop('Species')
test_y = test.pop('Species')

# Define the Input Function
def input_fn(features, labels, training=True, batch_size=256):
    # Convert the inputs into a Dataset
    dataset = tf.data.Dataset.from_tensor_slices((dict(features), labels))

    # Shuffle and repeat if in training mode
    if training:
        dataset = dataset.shuffle(1000).repeat()

    return dataset.batch(batch_size)

# Feature columns describe how to use the input
my_feature_columns = []
for key in train.keys():
    my_feature_columns.append(tf.feature_column.numeric_column(key=key))


# Build a DNN with 2 hidden layers with 30 and 10 hidden nodes each.
classifier = tf.estimator.DNNClassifier(feature_columns=my_feature_columns,
    # Two hidden laters of 30 and 10 nodes respectively.
    hidden_units=[30, 10],
    # The model must choose between 3 classes.
    n_classes = 3)

classifier.train(
    input_fn=lambda: input_fn(train, train_y, training=True),
    steps=5000)

#eval_result = classifier.evaluate(input_fn=lambda: input_fn(test, test_y, training=False))

#print(f'\nTest set accuracy: {format(eval_result["accuracy"], "0.3f")}\n')

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
    print(pred_dict)
    class_id = pred_dict['class_ids'][0]
    probability = pred_dict['probabilities'][class_id]

    print(f'Prediction is "{SPECIES[class_id]}" ({format(100 * probability, ".3f")})')