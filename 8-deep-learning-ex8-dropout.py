# %% [markdown]
# <h1> Dropout Excercise </h1>
# In the following exercise, you will have a chance to try out different dropout rates (p), and will be able to check which had the greatest effect on the model, in terms of performance.

# %% [code]
# Imports
import tensorflow as tf
from tensorflow import keras

# %% [code]
# Meet fashion MNIST - you will get familiar with this dataset soon...
(train_images, train_labels),(test_images, test_labels) = keras.datasets.fashion_mnist.load_data()
train_images = train_images /  255.0
test_images = test_images / 255.0
validation_images = train_images[:5000]
validation_labels = train_labels[:5000]

# %% [code]
# Another cool way to define networks is using Classes
# this makes the process pretty much automatic, and not dependent on some static values
### TODO: See the dropout rate below (0.2). 
# 1.Make sure to inject it as a parameter, so when we instantiate a model,
# we will be able to define its droput rate accordingly.
# 2. write a for loop of 3 models (no need for more),
# 2.1 each one of the models should be running on the same optimizers and compile as below
# 2.2 each one of the models should be running on the same optimizers and compile as below
# 3. Once the for loop ended, make sure to plot graphs of training performance + test performance.
# For that, you can use one of the old ex. we had.
# Dropout values can be randomly picked (ranging between 0 - 1)
### In other words, what I'm asking is a Hyperparameter random search, so you can implement this with some
# python library you know.
class CustomModel(keras.Model):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.input_layer = keras.layers.Flatten(input_shape=(28,28))
        self.hidden1 = keras.layers.Dense(200, activation='relu')
        self.hidden2 = keras.layers.Dense(100, activation='relu')
        self.hidden3 = keras.layers.Dense(60, activation='relu')
        self.output_layer = keras.layers.Dense(10, activation='softmax')
        self.dropout_layer = keras.layers.Dropout(rate=0.2)
    
    def call(self, input, training=None):
        input_layer = self.input_layer(input)
        input_layer = self.dropout_layer(input_layer)
        hidden1 = self.hidden1(input_layer)
        hidden1 = self.dropout_layer(hidden1, training=training)
        hidden2 = self.hidden2(hidden1)
        hidden2 = self.dropout_layer(hidden2, training=training)
        hidden3 = self.hidden3(hidden2)
        hidden3 = self.dropout_layer(hidden3, training=training)
        output_layer = self.output_layer(hidden3)
        return output_layer

# %% [code]
model = CustomModel()
sgd = keras.optimizers.SGD(lr=0.01)
model.compile(loss="sparse_categorical_crossentropy", optimizer=sgd, metrics=["accuracy"])

# %% [code]
model.fit(train_images, train_labels, epochs=60, validation_data=(validation_images, validation_labels))

# %% [code]
model.evaluate(test_images, test_labels)