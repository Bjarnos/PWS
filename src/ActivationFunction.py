# source: https://www.geeksforgeeks.org/deep-learning/building-artificial-neural-networks-ann-from-scratch/
import numpy as np

class ActivationFunction:
    @staticmethod
    def calculate(x: np.ndarray) -> np.ndarray: return np.empty((0, 0))

    @staticmethod
    def derivative(x: np.ndarray) -> np.ndarray: return np.empty((0, 0))

# ReLU: the activation function for the hidden layer
class ReLU(ActivationFunction):
    @staticmethod
    def calculate(x: np.ndarray) -> np.ndarray:
        return np.maximum(0, x)
    
    @staticmethod
    def derivative(x: np.ndarray) -> np.ndarray:
        return (x > 0).astype(np.float64)

# Sigmoid: another activation function for the hidden layer
class Sigmoid(ActivationFunction):
    @staticmethod
    def calculate(x: np.ndarray) -> np.ndarray:
        return 1 / (1 + np.exp(-x))
    
    @staticmethod
    def derivative(x: np.ndarray) -> np.ndarray:
        v = Sigmoid.calculate(x)
        return v * (1 - v)

# Softmax: calculates probability for every neuron based on its value
class Softmax(ActivationFunction):
    @staticmethod
    def calculate(x: np.ndarray) -> np.ndarray:
        # Subtract the maximum value from each input vector,
        # this prevents overflow when calculating exp(x)
        shifted_x = x - np.max(x, axis=-1, keepdims=True)
        
        exp_x = np.exp(shifted_x)
        sum_exp_x = np.sum(exp_x, axis=-1, keepdims=True)
        
        # Return normalized probabilities
        return exp_x / sum_exp_x
