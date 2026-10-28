# EXERCISE 4.5: AlexNet Dense Layer Parameter Comparison
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


def create_alexnet(dense_units=4096, num_classes=17):
  """Builds AlexNet with configurable dense layer units."""
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

  # Flatten Layer (Outputs a vector of size 5 * 5 * 256 = 6,400)
  model.add(Flatten())

  # 1st Dense Layer (Modified units)
  model.add(Dense(dense_units, activation='relu'))
  model.add(Dropout(0.4))

  # 2nd Dense Layer (Modified units)
  model.add(Dense(dense_units, activation='relu'))
  model.add(Dropout(0.4))

  # Output Layer
  model.add(Dense(num_classes, activation='softmax'))

  return model


# Build models with different dense layer units
model_4096 = create_alexnet(dense_units=4096)  # Original
model_2048 = create_alexnet(dense_units=2048)  # Modified (Option A)
model_1024 = create_alexnet(dense_units=1024)  # Modified (Option B)

# Display parameter comparison
print(
    f"{'Dense Layer Units':<20} | {'Total Parameters':<20} | {'Parameter'}"
    " Reduction"
)
print("-" * 65)

base_params = model_4096.count_params()
print(f"{'4096 Units (Original)':<20} | {base_params:<20,d} | Base Model")

params_2048 = model_2048.count_params()
red_2048 = ((base_params - params_2048) / base_params) * 100
print(
    f"{'2048 Units':<20} | {params_2048:<20,d} |"
    f" -{red_2048:.2f}%"
)

params_1024 = model_1024.count_params()
red_1024 = ((base_params - params_1024) / base_params) * 100
print(
    f"{'1024 Units':<20} | {params_1024:<20,d} |"
    f" -{red_1024:.2f}%"
)