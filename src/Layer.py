# source: https://github.com/joohei/mnist-from-scratch/blob/main
import numpy as np

class Layer():
    def forward(): pass
    def backward(): pass

# class Flatten(Layer):
#     def __init__(self, input_shape):
#         pass

class Dense(Layer):
    def __init__(self, num_neurons, activation):
        self.size = num_neurons
        self.activation_func = activation
        self.inputs = None
        self.outputs = None
        # self.weights = np.random.normal(0, 0.01, (output_size, input_size)) # over nadenken of we ook de output size willen hebben

    def forward(self, inputs):
        self.inputs = inputs
        self.outputs=self.weights * self.inputs + self.biases

    def backward(self, output_gradient, learn_rate, momentum, clip_value):
        pass
