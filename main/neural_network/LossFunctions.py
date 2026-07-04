"""
Some functions below have an `epsilon` parameter. This is a
value very close to zero, which you can make bigger or
smaller if needed.
"""

import jax.numpy as np

class LossFunction:
    """
    The parent class for all loss functions.

    <em>You should not use this class directly, but rather
    one of the child classes. It is only exported to use
    for typing.</em>
    """

    def calculate(self, predicted: np.ndarray, expected: np.ndarray) -> np.ndarray:
        "@private"
        return np.empty((0, 0)) # pyright: ignore[reportUnknownMemberType]
    
    def derivative(self, predicted: np.ndarray, expected: np.ndarray) -> np.ndarray:
        "@private"
        return np.empty((0, 0)) # pyright: ignore[reportUnknownMemberType]

class MeanSquaredError(LossFunction):
    """
    A loss function which calculates the average of the
    squares of the errors (predicted - expected).
    """

    def calculate(self, predicted: np.ndarray, expected: np.ndarray):
        r'''$$\text{MSE}=\sum_{i=1}^{n}(y_i-\hat{y_i})^2$$'''
        return np.sum(np.square(expected - predicted))

    def derivative(self, predicted: np.ndarray, expected: np.ndarray):
        r'''$$\text{MSE}'=2(\hat{y_i}-y_i)$$'''
        return 2 * (predicted - expected)
    
class MeanAbsoluteError(LossFunction):
    """
    A loss function which calculates the average of the
    absolutes of the errors.
    """

    def calculate(self, predicted: np.ndarray, expected: np.ndarray):
        r'''$$\text{MAE}=\sum_{i=1}^{n}|y_i-\hat{y_i}|$$'''
        return np.sum(np.abs(expected - predicted))

    def derivative(self, predicted: np.ndarray, expected: np.ndarray):
        r'''$$\text{MAE}'=\begin{cases}+1, & \text{if } y_i < \hat{y_i} \\\\-1, & \text{if } y_i > \hat{y_i}\end{cases}$$'''
        return np.sign(predicted - expected)
    
class CategorialCrossEntropy(LossFunction):
    """
    A loss function for networks with multiple categories.
    """

    def __init__(self, epsilon: float = 1e-9):
        self.epsilon = epsilon

    def calculate(self, predicted: np.ndarray, expected: np.ndarray):
        r'''$$\text{CCE}=-\sum_{i=1}^{n}y_i\ln(\hat{y_i})$$'''
        return -np.sum(expected * np.log(np.clip(predicted, self.epsilon, 1 - self.epsilon)))

    def derivative(self, predicted: np.ndarray, expected: np.ndarray):
        r'''$$\text{CCE}'=\hat{y_i} - y_i$$'''
        return predicted - expected
    
class KLDivergence(LossFunction):
    """
    A loss function which calculates the distance between
    two probability distributions.
    """

    def __init__(self, epsilon: float = 1e-9):
        self.epsilon = epsilon

    def calculate(self, predicted: np.ndarray, expected: np.ndarray):
        r'''$$\text{KLD}=\sum_{i=1}^{n}y_i\ln(\frac{y_i}{\hat{y_i}})$$'''
        return np.sum(expected * np.log(np.clip(expected / (predicted + self.epsilon), self.epsilon, 1e9)))

    def derivative(self, predicted: np.ndarray, expected: np.ndarray):
        r'''$$\text{KLD}'=\hat{y_i} - y_i$$'''
        return predicted - expected

__all__ = [
    "LossFunction", "MeanSquaredError", "MeanAbsoluteError",
    "CategorialCrossEntropy", "KLDivergence"
    ]
