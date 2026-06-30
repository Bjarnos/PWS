import numpy as np

class LossFunction:
    def __init__(self):
        pass
    def calculate(self, predicted, expected):
        pass
    def derivative(self, predicted, expected):
        pass


class MeanSquaredError(LossFunction):
    def __init__(self):
        super().__init__()

    def calculate(self, predicted, expected):
        return np.sum(np.square(expected - predicted))

    def derivative(self, predicted, expected):
        return 2 * (predicted - expected)
    
class MeanAbsoluteError(LossFunction):
    def __init__(self):
        super().__init__()

    def calculate(self, predicted, expected):
        return np.sum(np.abs(expected - predicted))

    def derivative(self, predicted, expected):
        return 1
    

class CategorialCrossEntropy(LossFunction):
    def __init__(self, epsilon):
        super().__init__()
        self.epsilon = epsilon

    def calculate(self, predicted, expected):
        return -np.sum(expected * np.log(np.clip(predicted, self.epsilon, 1 - self.epsilon)))

    def derivative(self, predicted, expected):
        return predicted - expected