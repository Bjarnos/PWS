# source: https://www.geeksforgeeks.org/deep-learning/building-artificial-neural-networks-ann-from-scratch/
import numpy as np

# ReLU: the activation function for the hidden layer
def ReLU(x: int) -> int:
    return np.maximum(0, x)

# derivative ReLU
def DReLU(x: int) -> int:
    return (x > 0).astype(int)

ReLU.derivative = DReLU

# Sigmoid: another activation function for the hidden layer
def Sigmoid(x: int) -> int:
    return 1 / (1 + np.exp(-x))

# derivative Sigmoid
def DSigmoid(x: int) -> int:
    return Sigmoid(x) * (1 - Sigmoid(x))

Sigmoid.derivative = DSigmoid

# Softmax: calculates probability for every neuron based on its value
def Softmax(x: np.ndarray) -> np.ndarray:
    # Subtract the maximum value from each input vector,
    # this prevents overflow when calculating exp(x)
    shifted_x = x - np.max(x, axis=-1, keepdims=True)
    
    exp_x = np.exp(shifted_x)
    sum_exp_x = np.sum(exp_x, axis=-1, keepdims=True)
    
    # Return normalized probabilities
    return exp_x / sum_exp_x
