# source: https://github.com/joohei/mnist-from-scratch/blob/main
import numpy as np

class Layer():
    def forward(): pass
    def backward(): pass

# class Flatten(Layer):
#     def __init__(self, input_shape):
#         pass

class Dense(Layer):
    def __init__(self, input_size, output_size, activation):
        self.input_size = input_size
        self.output_size = output_size
        self.activation_func = activation
        self.inputs = None
        self.outputs = None
        self.weights = np.random.randn(self.output_size, self.input_size) * np.sqrt(2.0 / self.input_size)
        self.biases = np.zeros(self.output_size)

    def forward(self, inputs):
        self.inputs = inputs
        x = np.dot(self.weights, inputs) + self.biases
        self.output = self.activation_func(x)
        return self.output

    def backward(self, output_gradient, learn_rate, clip_value):
        weight_gradient = np.mean(output_gradient @ np.swapaxes(self.inputs, 1, 2), axis=0)
        bias_gradient = np.mean(output_gradient, axis=0)

        # clip gradients
        weight_gradient = np.clip(weight_gradient, -clip_value, clip_value)
        bias_gradient = np.clip(bias_gradient, -clip_value, clip_value)

        self.weights = self.weights - weight_gradient * learn_rate
        self.biases = self.biases - bias_gradient * learn_rate

        return self.weights.T @ output_gradient
