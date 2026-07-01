# source: https://github.com/joohei/mnist-from-scratch/blob/main
from ActivationFunction import ActivationFunction
import numpy as np

class Layer():
    input_size: int
    output_size: int
    activation: type[ActivationFunction]

    inputs: np.ndarray
    z: np.ndarray
    weights: np.ndarray
    biases: np.ndarray
    prev_weight_momentum: np.ndarray
    prev_bias_momentum: np.ndarray

    def forward(self, inputs: np.ndarray) -> np.ndarray: return np.empty((0, 0))
    def backward(self, output_gradient: np.ndarray, learn_rate: float, momentum: float, clip_value: float, is_last: bool = False) -> np.ndarray: return np.empty((0, 0))


class Dense(Layer):
    def __init__(self, input_size: int, activation: type[ActivationFunction]):
        self.input_size = input_size
        self.activation = activation

        self.inputs = np.empty((0, 0))
        self.z = np.empty((0, 0))
        self.weights = np.empty((0, 0))
        self.biases = np.empty((0, 0))
        self.prev_weight_momentum = np.empty((0, 0))
        self.prev_bias_momentum = np.empty((0, 0))
    
    def forward(self, inputs: np.ndarray) -> np.ndarray:
        self.inputs = inputs
        self.z = self.inputs @ self.weights + self.biases
        return self.activation.calculate(self.z)
    
    def backward(self, output_gradient: np.ndarray, learn_rate: float, momentum: float, clip_value: float, is_last: bool = False) -> np.ndarray:
        dZ = output_gradient * (1.0 if is_last else self.activation.derivative(self.z))
        
        
        batch_size = self.inputs.shape[0]
        weight_gradient = (self.inputs.T @ dZ) / batch_size
        bias_gradient = np.mean(dZ, axis=0)

        if clip_value:
            weight_gradient = np.clip(weight_gradient, -clip_value, clip_value)
            bias_gradient = np.clip(bias_gradient, -clip_value, clip_value)
        
        next_output_gradient = dZ @ self.weights.T

        weight_momentum = momentum * self.prev_weight_momentum - learn_rate * weight_gradient
        bias_momentum = momentum * self.prev_bias_momentum - learn_rate * bias_gradient

        self.prev_weight_momentum = weight_momentum
        self.prev_bias_momentum = bias_momentum

        self.weights += weight_momentum.astype(np.float64)
        self.biases += bias_momentum.astype(np.float64)
        return next_output_gradient
