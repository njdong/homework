import numpy as np


class LinearRegression:
    """
    A linear regression model that uses analytical solutions.
    """

    w: np.ndarray
    b: float

    def __init__(self):
        """
        attributes w and b initialized to 0
        """
        self.w = 0
        self.b = 0

    def fit(self, X: np.ndarray, y: np.ndarray) -> np.ndarray:
        """
        lin algebra fit
        """
        if np.linalg.det(X.T @ X) != 0:
            param = np.linalg.inv(X.T @ X) @ X.T @ y
            self.w = param
            # self.b = param[1]
        else:
            print("LinAlgError. Matrix is Singular. No analytical solution.")

        return param

    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        predict with matrix multiplication
        """
        y_pred = self.w * X + self.b
        return y_pred


class GradientDescentLinearRegression(LinearRegression):
    """
    A linear regression model that uses gradient descent to fit the model.
    """

    def fit(
        self, X: np.ndarray, y: np.ndarray, lr: float = 0.01, epochs: int = 1000
    ) -> None:
        """
        update fit for the number of epochs
        """

        m = len(y)

        for i in range(epochs):
            y_pred = self.predict(X)
            self.w -= lr * ((1 / m) * np.sum(y_pred - y))
            self.b -= lr * ((1 / m) * np.sum(y_pred - y) * X)

    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Predict the output for the given input.

        Arguments:
            X (np.ndarray): The input data.

        Returns:
            np.ndarray: The predicted output.

        """

        y_pred = self.w * X + self.b

        return y_pred
