import time
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from tensorflow.keras.layers import Dense
from tensorflow.keras.models import Sequential

# Prepare Dataset
whole_data = load_breast_cancer()
X_data = whole_data.data
y_data = whole_data.target

X_train, X_test, y_train, y_test = train_test_split(
    X_data, y_data, test_size=0.3, random_state=7
)
features = X_train.shape[1]


def build_and_train_model(neuron_count):
  """Builds, trains, and evaluates MLP model with specified neuron count."""
  model = Sequential([
      Dense(neuron_count, activation='relu', input_shape=(features,)),
      Dense(neuron_count, activation='relu'),
      Dense(1, activation='sigmoid'),  # Binary classification output
  ])

  model.compile(
      optimizer='adam', loss='binary_crossentropy', metrics=['accuracy']
  )

  # Measure training duration
  start_time = time.time()
  model.fit(
      X_train,
      y_train,
      batch_size=50,
      validation_split=0.2,
      epochs=100,
      verbose=0,
  )
  training_time = time.time() - start_time

  # Evaluate on test set
  loss, accuracy = model.evaluate(X_test, y_test, verbose=0)
  total_params = model.count_params()

  return {
      'neurons': neuron_count,
      'params': total_params,
      'time': training_time,
      'loss': loss,
      'accuracy': accuracy,
  }


# Run experiments
res_10 = build_and_train_model(10)
res_5 = build_and_train_model(5)

# Output Comparison Results
print(f"{'Metric':<25} | {'10 Neurons (Original)':<20} | {'5 Neurons (Modified)':<20}")
print("-" * 72)
print(f"{'Trainable Parameters':<25} | {res_10['params']:<20} | {res_5['params']:<20}")
print(f"{'Training Time (seconds)':<25} | {res_10['time']:<20.4f} | {res_5['time']:<20.4f}")
print(f"{'Test Loss':<25} | {res_10['loss']:<20.4f} | {res_5['loss']:<20.4f}")
print(f"{'Test Accuracy (%)':<25} | {res_10['accuracy']*100:<19.2f}% | {res_5['accuracy']*100:<19.2f}%")