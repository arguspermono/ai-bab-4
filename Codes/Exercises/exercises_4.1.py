from numpy import array, dot, exp, random

# Set up parameters (3rd column added as a bias term equal to 1) =====
inputs = array([[0, 0, 1], [0, 1, 1], [1, 0, 1], [1, 1, 1]])

# Target outputs for Logical OR: 0 OR 0 = 0, others = 1
outputs = array([[0, 1, 1, 1]]).T

random.seed(1)
# 3 weights: w1 (Input 1), w2 (Input 2), w3 (Bias)
weights = 2 * random.random((3, 1)) - 1


# Define a Single Neuron function ====================================
def neuron(inputs, weights):
  output = 1 / (1 + exp(-(dot(inputs, weights))))
  return output


# Train the Neuron ===================================================
for iteration in range(50000):
  output = neuron(inputs, weights)
  # Weight adjustment using the gradient descent delta rule
  weights += dot(inputs.T, (outputs - output) * output * (1 - output))

# Test the Neuron on all OR inputs ===================================
print("Logical OR Gate Predictions:")
test_cases = array([[0, 0, 1], [0, 1, 1], [1, 0, 1], [1, 1, 1]])

for x in test_cases:
  prediction = neuron(x, weights)[0]
  print(
      f"Input: [{x[0]}, {x[1]}] -> Output: {prediction:.5f} (Rounded:"
      f" {round(prediction)})"
  )