# Example 4.7 (Fixed for Keras 3)
from keras.layers import Conv2D, Dense, Dropout, Flatten, Input, MaxPooling2D
from keras.models import Sequential

# Create model
model = Sequential([
    Input(shape=(28, 28, 1)),
    Conv2D(32, (5, 5), activation='relu'),
    MaxPooling2D(),
    Dropout(0.2),
    Flatten(),
    Dense(128, activation='relu'),
    Dense(2, activation='softmax'),
])

# Compile model
model.compile(
    loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy']
)

print(model.summary())