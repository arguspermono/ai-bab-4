# EXERCISE 4.7: Experimenting with Filter Counts & Kernel Sizes
import matplotlib.pyplot as plt
from keras.datasets import mnist
from keras.layers import Conv2D, Dense, Dropout, Flatten, Input, MaxPooling2D
from keras.models import Sequential
from keras.utils import to_categorical

# Load and preprocess MNIST dataset
(X_train, y_train), (X_test, y_test) = mnist.load_data()
X_train = X_train.reshape((-1, 28, 28, 1)).astype('float32') / 255.0
X_test = X_test.reshape((-1, 28, 28, 1)).astype('float32') / 255.0
y_train = to_categorical(y_train)
y_test = to_categorical(y_test)
num_classes = y_test.shape[1]


def build_model(filters=32, kernel_size=(5, 5)):
  model = Sequential([
      Input(shape=(28, 28, 1)),
      Conv2D(filters, kernel_size, activation='relu'),
      MaxPooling2D(),
      Dropout(0.2),
      Flatten(),
      Dense(128, activation='relu'),
      Dense(num_classes, activation='softmax'),
  ])
  model.compile(
      loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy']
  )
  return model


# Instantiate models for Exercise 4.7
baseline_model = build_model(filters=32, kernel_size=(5, 5))  # Baseline
filters_64_model = build_model(filters=64, kernel_size=(5, 5))  # 64 Filters
kernel_3x3_model = build_model(filters=32, kernel_size=(3, 3))  # (3,3) Kernel