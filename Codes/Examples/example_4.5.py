# Example 4.5 AlexNet with Keras (Fixed & Updated for Keras 3)
import keras
from keras.layers import (
    BatchNormalization,
    Conv2D,
    Dense,
    Dropout,
    Flatten,
    Input,
    MaxPooling2D,
)
from keras.models import Sequential

# Create the AlexNet model
model = Sequential()

# Input Layer
model.add(Input(shape=(224, 224, 3)))

# 1st Convolutional Layer
model.add(
    Conv2D(
        filters=96,
        kernel_size=(11, 11),
        strides=(4, 4),
        activation='relu',
        padding='valid',
    )
)
model.add(BatchNormalization())
model.add(MaxPooling2D(pool_size=(3, 3), strides=(2, 2), padding='valid'))

# 2nd Convolutional Layer
model.add(
    Conv2D(
        filters=256,
        kernel_size=(5, 5),
        strides=(1, 1),
        activation='relu',
        padding='same',
    )
)
model.add(BatchNormalization())
model.add(MaxPooling2D(pool_size=(3, 3), strides=(2, 2), padding='valid'))

# 3rd Convolutional Layer
model.add(
    Conv2D(
        filters=384,
        kernel_size=(3, 3),
        strides=(1, 1),
        activation='relu',
        padding='same',
    )
)

# 4th Convolutional Layer
model.add(
    Conv2D(
        filters=384,
        kernel_size=(3, 3),
        strides=(1, 1),
        activation='relu',
        padding='same',
    )
)

# 5th Convolutional Layer
model.add(
    Conv2D(
        filters=256,
        kernel_size=(3, 3),
        strides=(1, 1),
        activation='relu',
        padding='same',
    )
)
model.add(MaxPooling2D(pool_size=(3, 3), strides=(2, 2), padding='valid'))

# Fully Connected Layers
model.add(Flatten())

# 1st Fully Connected Layer
model.add(Dense(4096, activation='relu'))
model.add(Dropout(0.4))

# 2nd Fully Connected Layer
model.add(Dense(4096, activation='relu'))
model.add(Dropout(0.4))

# Output Layer
model.add(Dense(17, activation='softmax'))

model.summary()

# Compile the model
model.compile(
    loss=keras.losses.categorical_crossentropy,
    optimizer='adam',
    metrics=['accuracy'],
)