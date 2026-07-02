import jax.numpy as np

class LossFunction:
    def calculate(self, predicted: np.ndarray, expected: np.ndarray) -> np.ndarray: return np.empty((0, 0)) # pyright: ignore[reportUnknownMemberType]
    def derivative(self, predicted: np.ndarray, expected: np.ndarray) -> np.ndarray: return np.empty((0, 0)) # pyright: ignore[reportUnknownMemberType]


class MeanSquaredError(LossFunction):
    def __init__(self):
        super().__init__()

    def calculate(self, predicted: np.ndarray, expected: np.ndarray):
        return np.sum(np.square(expected - predicted))

    def derivative(self, predicted: np.ndarray, expected: np.ndarray):
        return 2 * (predicted - expected)
    
class MeanAbsoluteError(LossFunction):
    def __init__(self):
        super().__init__()

    def calculate(self, predicted: np.ndarray, expected: np.ndarray):
        return np.sum(np.abs(expected - predicted))

    def derivative(self, predicted: np.ndarray, expected: np.ndarray):
        return np.sign(predicted - expected)
    
class CategorialCrossEntropy(LossFunction):
    def __init__(self, epsilon: float = 1e-9):
        super().__init__()
        self.epsilon = epsilon

    def calculate(self, predicted: np.ndarray, expected: np.ndarray):
        return -np.sum(expected * np.log(np.clip(predicted, self.epsilon, 1 - self.epsilon)))

    def derivative(self, predicted: np.ndarray, expected: np.ndarray):
        return predicted - expected
    