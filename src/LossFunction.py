import numpy as np

class LossFunction:
    def calculate(self, predicted: np.ndarray, expected: np.ndarray) -> np.ndarray: return np.empty((0, 0))
    def derivative(self, predicted: np.ndarray, expected: np.ndarray) -> np.ndarray: return np.empty((0, 0))


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
        return (predicted > expected).astype(np.int64) - (predicted < expected).astype(np.int64)
    
class CategorialCrossEntropy(LossFunction):
    def __init__(self, epsilon: float):
        super().__init__()
        self.epsilon = epsilon

    def calculate(self, predicted: np.ndarray, expected: np.ndarray):
        return -np.sum(expected * np.log(np.clip(predicted, self.epsilon, 1 - self.epsilon)))

    def derivative(self, predicted: np.ndarray, expected: np.ndarray):
        return predicted - expected
    