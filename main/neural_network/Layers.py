# source: https://github.com/joohei/mnist-from-scratch/blob/main
from .ActivationFunctions import ActivationFunction
from .Optimizers import Optimizer
import jax.numpy as np

class Layer():
    """
    The parent class for all layers.

    <em>You should not use this class directly, but rather
    one of the child classes. It is only exported to use
    for typing.</em>
    """

    input_size: int
    "The size of the layer before the current"
    output_size: int
    "The size of the current layer"
    activation: ActivationFunction
    "A reference to the activation function of this layer"

    inputs: np.ndarray
    z: np.ndarray
    weights: np.ndarray
    biases: np.ndarray
    weight_momentum: np.ndarray
    bias_momentum: np.ndarray
    weight_varience: np.ndarray
    bias_varience: np.ndarray
    acc_w_grad: np.ndarray
    acc_b_grad: np.ndarray
    
    def forward(self, inputs: np.ndarray) -> np.ndarray:
        "@private"
        return np.empty((0, 0)) # pyright: ignore[reportUnknownMemberType]
    
    def backward(self, output_gradient: np.ndarray, optimizer: Optimizer, clip_value: float, is_last: bool = False) -> np.ndarray:
        "@private"
        return np.empty((0, 0)) # pyright: ignore[reportUnknownMemberType]

class Dense(Layer):
    """
    A fully connected layer where each neuron of the current
    layer is connected to each neuron of the previous layer

    
    """

    input_size: int
    output_size: int
    activation: ActivationFunction

    inputs: np.ndarray
    z: np.ndarray
    weights: np.ndarray
    biases: np.ndarray
    weight_momentum: np.ndarray
    bias_momentum: np.ndarray
    weight_varience: np.ndarray
    bias_varience: np.ndarray
    acc_w_grad: np.ndarray
    acc_b_grad: np.ndarray

    def __init__(self, input_size: int, activation: ActivationFunction):
        self.input_size = input_size
        self.activation = activation

        self.inputs = np.empty((0, 0)) # pyright: ignore[reportUnknownMemberType]
        self.z = np.empty((0, 0)) # pyright: ignore[reportUnknownMemberType]
        self.weights = np.empty((0, 0)) # pyright: ignore[reportUnknownMemberType]
        self.biases = np.empty((0, 0)) # pyright: ignore[reportUnknownMemberType]
        self.weight_momentum = np.empty((0, 0)) # pyright: ignore[reportUnknownMemberType]
        self.bias_momentum = np.empty((0, 0)) # pyright: ignore[reportUnknownMemberType]
        self.weight_varience = np.empty((0, 0)) # pyright: ignore[reportUnknownMemberType]
        self.bias_varience = np.empty((0, 0)) # pyright: ignore[reportUnknownMemberType]
        self.acc_w_grad = np.empty((0, 0)) # pyright: ignore[reportUnknownMemberType]
        self.acc_b_grad = np.empty((0, 0)) # pyright: ignore[reportUnknownMemberType]
    
    def forward(self, inputs: np.ndarray) -> np.ndarray:
        "private function, used for a feed-forward pass"
        self.inputs = inputs
        self.z = self.inputs @ self.weights + self.biases
        return self.activation.calculate(self.z)
    
    def backward(self, output_gradient: np.ndarray, optimizer: Optimizer, clip_value: float, is_last: bool = False) -> np.ndarray:
        "private function, used for backpropagation"
        dZ = output_gradient * (1.0 if is_last else self.activation.derivative(self.z))
        
        batch_size = self.inputs.shape[0]
        weight_gradient = (self.inputs.T @ dZ) / batch_size
        bias_gradient = np.mean(dZ, axis=0)

        if clip_value:
            weight_gradient = np.clip(weight_gradient, -clip_value, clip_value)
            bias_gradient = np.clip(bias_gradient, -clip_value, clip_value)

        next_output_gradient = dZ @ self.weights.T

        self.weights, self.biases = optimizer.calculate(weight_gradient, bias_gradient, self.weight_varience, self.bias_varience, self.acc_w_grad, self.acc_b_grad, self.weights, self.biases, self.weight_momentum, self.bias_momentum)

        return next_output_gradient

__all__ = ["Layer", "Dense"]
