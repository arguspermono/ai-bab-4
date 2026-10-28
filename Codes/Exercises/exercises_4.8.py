# EXERCISE 4.8: Adding a Second Convolutional Block
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

# Deep CNN Architecture (2 Conv Blocks)
deep_model = Sequential([
    Input(shape=(28, 28, 1)),
    # Block 1
    Conv2D(32, (5, 5), activation='relu'),
    MaxPooling2D(),
    Dropout(0.2),
    # Block 2 (Added for Exercise 4.8)
    Conv2D(64, (3, 3), activation='relu'),
    MaxPooling2D(),
    Dropout(0.2),
    # Classifier
    Flatten(),
    Dense(128, activation='relu'),
    Dense(num_classes, activation='softmax'),
])

deep_model.compile(
    loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy']
)

# Train Deep CNN
deep_model.fit(
    X_train,
    y_train,
    validation_data=(X_test, y_test),
    epochs=10,
    batch_size=200,
)

# Evaluate Deep CNN
scores = deep_model.evaluate(X_test, y_test, verbose=0)
print(f'Deep CNN Error: {(100 - scores[1] * 100):.2f}%')
print(f'Deep CNN Accuracy: {(scores[1] * 100):.2f}%')