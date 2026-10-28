# EXERCISE 4.4: LeNet-5 Filter Comparison
from keras.layers import AveragePooling2D, Conv2D, Dense, Flatten, Input
from keras.models import Sequential


def create_lenet(conv1_filters=6, conv2_filters=16):
  """Builds a LeNet-5 model with configurable conv filter sizes."""
  model = Sequential([
      Input(shape=(32, 32, 1)),
      Conv2D(filters=conv1_filters, kernel_size=(3, 3), activation='relu'),
      AveragePooling2D(pool_size=(2, 2)),
      Conv2D(filters=conv2_filters, kernel_size=(3, 3), activation='relu'),
      AveragePooling2D(pool_size=(2, 2)),
      Flatten(),
      Dense(units=120, activation='relu'),
      Dense(units=84, activation='relu'),
      Dense(units=10, activation='softmax'),
  ])
  return model


# Build models with different filter counts
model_original = create_lenet(conv1_filters=6, conv2_filters=16)  # Original
model_doubled = create_lenet(conv1_filters=12, conv2_filters=32)  # Doubled
model_halved = create_lenet(conv1_filters=3, conv2_filters=8)  # Halved

# Print comparison summary
base_params = model_original.count_params()
doubled_params = model_doubled.count_params()
halved_params = model_halved.count_params()

print(
    f"{'Filter Config (Conv1, Conv2)':<30} | {'Total Parameters':<18} |"
    " Relative Change"
)
print("-" * 72)
print(
    f"{'(6, 16) - Original':<30} | {base_params:<18,d} | Base Model"
)
print(
    f"{'(12, 32) - Doubled':<30} | {doubled_params:<18,d} |"
    f" +{((doubled_params - base_params) / base_params) * 100:.2f}%"
)
print(
    f"{'(3, 8) - Halved':<30} | {halved_params:<18,d} |"
    f" {((halved_params - base_params) / base_params) * 100:.2f}%"
)