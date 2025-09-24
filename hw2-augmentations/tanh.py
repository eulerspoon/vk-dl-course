import numpy as np

class Tanh:
    def __init__(self):
        pass

    def forward(self, X):
        self.X = X
        self.tanh = (np.exp(X) - np.exp(-X)) / (np.exp(X) + np.exp(-X))
        return self.tanh

    def backward(self, dLdY):
        return (1 - self.tanh * self.tanh) * dLdY

    def step(self, learning_rate):
        pass
