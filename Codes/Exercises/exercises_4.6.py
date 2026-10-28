# EXERCISE 4.6: Adding a second Conv2D, MaxPooling2D, and Dropout layer
from keras.layers import Conv2D, Dense, Dropout, Flatten, Input, MaxPooling2D
from keras.models import Sequential


# 1. Original Example 4.7 Model
def build_original_model():
  return Sequential(
      [
          Input(shape=(28, 28, 1)),
          Conv2D(32, (5, 5), activation='relu'),
          MaxPooling2D(),
          Dropout(0.2),
          Flatten(),
          Dense(128, activation='relu'),
          Dense(2, activation='softmax'),
      ],
      name='Original_Model',
  )


# 2. Modified Model (Added Conv2D + MaxPooling2D + Dropout block)
def build_modified_model():
  return Sequential(
      [
          Input(shape=(28, 28, 1)),
          # First Conv Block
          Conv2D(32, (5, 5), activation='relu'),
          MaxPooling2D(),
          Dropout(0.2),
          # Second Conv Block (Added)
          Conv2D(64, (3, 3), activation='relu'),
          MaxPooling2D(),
          Dropout(0.2),
          # Classifier Block
          Flatten(),
          Dense(128, activation='relu'),
          Dense(2, activation='softmax'),
      ],
      name='Modified_Model',
  )


# Instantiate models
original_model = build_original_model()
modified_model = build_modified_model()

# Parameter Comparison Summary
orig_params = original_model.count_params()
mod_params = modified_model.count_params()

print(f"{'Model Version':<25} | {'Total Parameters':<18}")
print("-" * 46)
print(f"{'Original (1 Conv Block)':<25} | {orig_params:<18,d}")
print(f"{'Modified (2 Conv Blocks)':<25} | {mod_params:<18,d}")
print(f"{'Net Parameter Change':<25} | {mod_params - orig_params:<18,d}")
