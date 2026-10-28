from sklearn.neural_network import MLPClassifier

# Updated X to have 3 inputs (features) per sample
X = [
    [0.0, 0.0, 0.0],
    [1.0, 1.0, 1.0],
    [0.0, 1.0, 0.0],
    [1.0, 0.0, 1.0]
]
y = [0, 1, 1, 1]

clf = MLPClassifier(
    solver='lbfgs', alpha=1e-5, hidden_layer_sizes=(5, 2), random_state=1
)
clf.fit(X, y)

# Updated test predictions to match 3 input features
print("Predictions:", clf.predict([[2.0, 2.0, 2.0], [-1.0, -2.0, -3.0]]))
print("Layer weight shapes:", [coef.shape for coef in clf.coefs_])
