import numpy as np

from ActivationFunction import *
from Layer import Layer

class NeuralNetwork:
    def __init__(self, layers: list[Layer]):
        self.layers = layers

    def forward(self, inputs):
        for layer in self.layers:
            inputs = layer.forward(inputs)

        return inputs

    def train_epoch(self, input_data, output_data):
        total_loss = 0

        learning_rate = 0.01
        clip_value = 5.0

        for i in range(len(input_data)):
            target_output = output_data[i]
            output = self.forward(input_data[i])

            output_clipped = np.clip(output, 1e-15, 1.0 - 1e-15)
            total_loss -= np.sum(target_output * np.log(output_clipped))

            gradient = output - target_output 
            
            for layer in reversed(self.layers):
                gradient = layer.backward(gradient, learning_rate, clip_value, layer == self.layers[-1])

        return total_loss / len(input_data)

    def train_model(self, input_data, output_data, epochs: int = 5):
        # Generate weight and bias lists
        for i in range(len(self.layers)):
            current_layer = self.layers[i]
            input_size = current_layer.input_size
            
            if i == len(self.layers) - 1:
                output_size = len(output_data[0]) 
            else:
                output_size = self.layers[i+1].input_size

            # Kaiming Initialization
            current_layer.output_size = output_size
            current_layer.weights = np.random.randn(output_size, input_size) * np.sqrt(2.0 / input_size)
            current_layer.biases = np.zeros(output_size)
        
        # Train epochs
        for epoch in range(epochs):
            loss = self.train_epoch(input_data, output_data)
            print(f"Epoch {epoch + 1}/{epochs} - Loss: {loss:.4f}")

        print("Done!")

    def test_model(self, input_data):
        return self.forward(input_data)
 