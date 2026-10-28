# Example 4.4 LeNet-5 Keras (Keras 3 Compatible)
from keras.layers import AveragePooling2D, Conv2D, Dense, Flatten, Input
from keras.models import Sequential

model = Sequential([
    # Menggunakan Input(shape=...) untuk menghindari UserWarning
    Input(shape=(32, 32, 1)),
    Conv2D(filters=6, kernel_size=(3, 3), activation='relu'),
    # Menambahkan argumen pool_size=(2, 2)
    AveragePooling2D(pool_size=(2, 2)),
    Conv2D(filters=16, kernel_size=(3, 3), activation='relu'),
    AveragePooling2D(pool_size=(2, 2)),
    Flatten(),
    Dense(units=120, activation='relu'),
    Dense(units=84, activation='relu'),
    Dense(units=10, activation='softmax'),
])

model.summary()