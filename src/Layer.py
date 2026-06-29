# source: https://github.com/joohei/mnist-from-scratch/blob/main
import numpy as np

class Layer():
    def forward(): pass
    def backward(): pass

# class Flatten(Layer):
#     def __init__(self, input_shape):
#         pass

class Dense(Layer):
    def __init__(self, input_size, activation):
        self.input_size = input_size
        self.activation_func = activation

        self.inputs = None
        self.z = None
        self.weights = None
        self.biases = None

    def forward(self, inputs):
        self.inputs = inputs
        self.z = np.dot(self.weights, inputs) + self.biases
        return self.activation_func(self.z)

    def backward(self, output_gradient, learn_rate, clip_value, is_last = False):
        dZ = output_gradient * (1.0 if is_last else self.activation_func.derivative(self.z))
        
        weight_gradient = np.outer(dZ, self.inputs)
        bias_gradient = dZ

        if clip_value:
            weight_gradient = np.clip(weight_gradient, -clip_value, clip_value)
            bias_gradient = np.clip(bias_gradient, -clip_value, clip_value)

        next_output_gradient = np.dot(self.weights.T, dZ)

        self.weights -= learn_rate * weight_gradient
        self.biases -= learn_rate * bias_gradient

        return next_output_gradient
