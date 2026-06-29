import numpy as np

from ActivationFunction import *
from Layer import Layer

def forward_propagation(input, weights, biases, layers):
    cache = [input]
    current_input = input
    
    for l in range(len(layers)):
        x = np.dot(weights[l], current_input) + biases[l]
        a = layers[l].activation_func(x)
        cache.append(a)
        current_input = a
        
    return current_input, cache

def backward_propagation(output, targets, cache, weights, layers):
    weight_gradients = [None] * len(weights)
    bias_gradients = [None] * len(weights)
    
    dZ = output - targets # dit mag alleen met als laatste layer een Softmax layer
    
    for l in reversed(range(len(weights))):
        a_prev = cache[l]
        
        weight_gradients[l] = np.outer(dZ, a_prev)
        bias_gradients[l] = dZ
        
        if l > 0:
            d_activation = layers[l-1].activation_func.derivative(a_prev)
            dZ = np.dot(weights[l].T, dZ) * d_activation
            
    return weight_gradients, bias_gradients

class NeuralNetwork:
    def __init__(self, layers: list[Layer]):
        self.layers = layers

    def train_epoch(self, weights, biases, input_data, output_data):
        total_loss = 0

        for i in range(len(input_data)):
            target_output = output_data[i]
            output, cache = forward_propagation(input_data[i], weights, biases, self.layers)

            output_clipped = np.clip(output, 1e-15, 1.0 - 1e-15)
            total_loss -= np.sum(target_output * np.log(output_clipped))

            weight_gradients, bias_gradients = backward_propagation(output, target_output, cache, weights, self.layers)

            learning_rate = 0.01 # kan later veranderd worden
            for j in range(len(weights)):
                weights[j] -= learning_rate * weight_gradients[j]
                biases[j] -= learning_rate * bias_gradients[j]

        return total_loss / len(input_data)

    def train_model(self, input_data, output_data, epochs: int = 5):
        weights = []
        biases = []
        
        # Generate weight and bias lists
        for i in range(len(self.layers)):
            input_dim = self.layers[i].size
            
            if i == len(self.layers) - 1:
                output_dim = len(output_data[0]) 
            else:
                output_dim = self.layers[i+1].size

            # Kaiming Initialization
            weights.append(np.random.randn(output_dim, input_dim) * np.sqrt(2.0 / input_dim))
            biases.append(np.zeros(output_dim))
        
        # Train epochs
        for epoch in range(epochs):
            loss = self.train_epoch(weights, biases, input_data, output_data)
            print(f"Epoch {epoch + 1}/{epochs} - Loss: {loss:.4f}")

        print("Done!")

    def test_model(self, input_data):
        pass
 