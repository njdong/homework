import numpy as np


class LinearRegression:
    w: np.ndarray
    b: float

    def __init__(self):
        self.w = 0
        self.b = 0

    def fit(self, X, y):
        if np.linalg.det(X.T @ X) != 0:
            self.w, self.b = np.linalg.inv(X.T @ X) @ X.T @ y
        else:
            print("LinAlgError. Matrix is Singular. No analytical solution.")
        return self.w, self.b

    def predict(self, X):
        y_pred = self.w * X + self.b
        return y_pred


class GradientDescentLinearRegression(LinearRegression):
    """
    A linear regression model that uses gradient descent to fit the model.
    """

    def fit(
        self, X: np.ndarray, y: np.ndarray, lr: float = 0.01, epochs: int = 1000
    ) -> None:
        m = len(y)

        for x in range(epochs):
            y_pred = self.predict()
            self.w += lr * ((1 / m) * np.sum(y_pred - y))
            self.b -= lr * ((1 / m) * np.sum(y_pred - y) * self.X)

    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Predict the output for the given input.

        Arguments:
            X (np.ndarray): The input data.

        Returns:
            np.ndarray: The predicted output.

        """

        y_pred = self.w * x + self.b

        return y_pred
