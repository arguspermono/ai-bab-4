# Example 4.2 Multiple Layer Perceptron
# https://scikit-learn.org/stable/modules/neural_networks_supervised.html
from sklearn.neural_network import MLPClassifier

X = [[0.0, 0.0], [1.0, 1.0], [0.0, 1.0], [1.0, 0.0]]
y = [0, 1, 1, 1]

# Fixed alpha value from '1e- 5' to '1e-5'
clf = MLPClassifier(
    solver='lbfgs', alpha=1e-5, hidden_layer_sizes=(5, 2), random_state=1
)
clf.fit(X, y)

# Cleaned up negative floats '[- 1., - 2.]' to '[-1., -2.]'
print("Predictions:", clf.predict([[2.0, 2.0], [-1.0, -2.0]]))
print("Layer weight shapes:", [coef.shape for coef in clf.coefs_])