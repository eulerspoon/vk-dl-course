import numpy as np

class LinearRegressorGD:
    """
    Линейная регрессия с использованием Gradient Descent
    """

    def __init__(self, learning_rate=0.01, n_iter=1000):
        """
        Конструктор класса

        Параметры:
            learning_rate (float): Скорость обучения
            n_iter (int): Количество итераций градиентного спуска
        """
        self.learning_rate = learning_rate
        self.n_iter = n_iter
        self.weights = None
        self.intercept = None
        self.reg_alpha = 0.001

    def fit(self, X, y):
        """
        Обучение модели на обучающей выборке с использованием
        градиентного спуска

        Параметры:
            X (np.ndarray): Матрица признаков размера (n_samples, n_features)
            y (np.ndarray): Вектор таргета длины n_samples
        """
        n_samples, n_features = X.shape
        self.weights = np.zeros(n_features)
        self.intercept = 0
        y = y.flatten() if len(y.shape) > 1 else y
        for _ in range(self.n_iter):
            predict = self.predict(X)
            diff = predict - y
            grad_weights = 2 * np.dot(X.T, diff) / n_samples + self.get_r2_reg()
            grad_intercept = 2 * np.mean(diff)
            self.weights -= self.learning_rate * grad_weights
            self.intercept -= self.learning_rate * grad_intercept
        return self

    def predict(self, X):
        """
        Получение предсказаний обученной модели

        Параметры:
            X (np.ndarray): Матрица признаков

        Возвращает:
            np.ndarray: Предсказание для каждого элемента из X
        """
        return np.dot(X, self.weights) + self.intercept
    
    def get_r2_reg(self):
        return 2 * self.reg_alpha * self.weights

    def get_params(self):
        """
        Возвращает обученные параметры модели
        """
        return self.weights + self.intercept
