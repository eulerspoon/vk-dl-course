import numpy as np

class MLPRegressor:
    def __init__(self, hidden_layer_sizes=(100,), learning_rate=0.01, n_iter=100):
        self.hidden_layer_sizes = hidden_layer_sizes
        self.hidden_layers_number = len(self.hidden_layer_sizes)
        self.learning_rate = learning_rate
        self.n_iter = n_iter

    def forward(self, X):
        self.linears = []
        self.activations = [X]

        for layer in range(self.hidden_layers_number):
            result = np.dot(self.activations[-1], self.weights[layer]) + self.biases[layer]
            self.linears.append(result)
            self.activations.append(self.sigmoid(result))

        result = np.dot(self.activations[-1], self.weights[-1]) + self.biases[-1]
        self.linears.append(result)
        self.activations.append(result)

        return result

    def backward(self, X, y):
        n_samples = X.shape[0]
        grad_weights = [None] * len(self.weights)
        grad_biases = [None] * len(self.biases)

        y_pred = self.forward(X)

        delta = 2 * (y_pred - y.reshape(-1, 1)) / n_samples

        for layer in range(len(self.weights) - 1, -1, -1):
            if layer == len(self.weights) - 1:
                grad_weights[layer] = np.dot(self.activations[layer].T, delta)
                grad_biases[layer] = np.sum(delta, axis=0, keepdims=True)
            else:
                delta = np.dot(delta, self.weights[layer + 1].T) * self.sigmoid_derivative(self.linears[layer])
                grad_weights[layer] = np.dot(self.activations[layer].T, delta)
                grad_biases[layer] = np.sum(delta, axis=0, keepdims=True)

        max_grad_norm = 10
        for i in range(len(grad_weights)):
            grad_norm = np.linalg.norm(grad_weights[i])
            if grad_norm > max_grad_norm:
                grad_weights[i] *= max_grad_norm / grad_norm

            grad_norm = np.linalg.norm(grad_biases[i])
            if grad_norm > max_grad_norm:
                grad_biases[i] *= max_grad_norm / grad_norm

        return grad_weights, grad_biases

    def fit(self, X, y):
        n_features = X.shape[1]
        all_sizes = [n_features] + list(self.hidden_layer_sizes) + [1]

        self.weights = []
        self.biases = []

        for i in range(len(all_sizes) - 1):
            sigma = np.sqrt(2.0 / (all_sizes[i] + all_sizes[i + 1]))
            self.weights.append(np.random.randn(all_sizes[i], all_sizes[i + 1]) * sigma)
            self.biases.append(np.zeros((1, all_sizes[i + 1])))

        for epoch in range(self.n_iter):
            grad_weights, grad_biases = self.backward(X, y)

            for i in range(len(self.weights)):
                self.weights[i] -= self.learning_rate * grad_weights[i]
                self.biases[i] -= self.learning_rate * grad_biases[i]

            if epoch % 100 == 0:
                y_pred = self.predict(X)
                loss = np.mean((y_pred - y.flatten())**2)
                print(f"Epoch {epoch}, loss: {loss}")

        return self

    def predict(self, X):
        result = X

        for layer in range(self.hidden_layers_number):
            result = self.sigmoid(np.dot(result, self.weights[layer]) + self.biases[layer])

        result = np.dot(result, self.weights[-1]) + self.biases[-1]

        return result.flatten()

    def sigmoid(self, z):
        return 1 / (1 + np.exp(-z))

    def sigmoid_derivative(self, z):
        s = self.sigmoid(z)
        return s * (1 - s)

    def get_params(self):
        return self.weights, self.biases
