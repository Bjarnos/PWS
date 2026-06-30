# source: https://github.com/joohei/mnist-from-scratch
import numpy as np
import joblib
import time

from ActivationFunction import *
from Layer import Layer

class Batch:
    def __init__(self, x_train, y_train):
        self.x = np.asarray(x_train)
        self.y = np.asarray(y_train)

def create_batches(x, y, size = 256):
    num_batches = int(np.ceil(x.shape[0] / size))
    x_train = np.array_split(x, num_batches)
    y_train = np.array_split(y, num_batches)
    return [Batch(_x, _y) for _x, _y in zip(x_train, y_train)]

class NeuralNetwork:
    def __init__(self, layers: list[Layer]):
        self.layers = layers
        self.learning_rate = 0.01
        self.clip_value = 5.0

    def init_weigths(self, output_layer_size):
        # Generate weight and bias lists
        for i in range(len(self.layers)):
            current_layer = self.layers[i]
            input_size = current_layer.input_size
            
            if i == len(self.layers) - 1:
                output_size = output_layer_size
            else:
                output_size = self.layers[i+1].input_size

            # Kaiming Initialization
            current_layer.output_size = output_size
            current_layer.weights = np.random.randn(input_size, output_size) * np.sqrt(2.0 / input_size)
            current_layer.biases = np.zeros(output_size)

    def forward(self, inputs):
        for layer in self.layers:
            inputs = layer.forward(inputs)

        return inputs

    # def train_epoch(self, input_data, output_data):
    #     total_loss = 0

    #     for i in range(len(input_data)):
    #         target_output = output_data[i]
    #         output = self.forward(input_data[i])

    #         output_clipped = np.clip(output, 1e-15, 1.0 - 1e-15)
    #         total_loss -= np.sum(target_output * np.log(output_clipped))

    #         gradient = output - target_output 
            
    #         for layer in reversed(self.layers):
    #             gradient = layer.backward(gradient, self.learning_rate, self.clip_value, layer == self.layers[-1])

    #     return total_loss / len(input_data)
    
    def train_epoch(self, batches: list[Batch]):
        total_loss = 0
        num_processed = 0

        for batch in np.random.permutation(batches):
            num_processed += batch.x.shape[0]
            outputs = self.forward(batch.x)
            output_clipped = np.clip(outputs, 1e-15, 1.0 - 1e-15)
            total_loss -= np.sum(batch.y * np.log(output_clipped))

            gradient = outputs - batch.y
            for layer in reversed(self.layers):
                gradient = layer.backward(gradient, self.learning_rate, self.clip_value, layer == self.layers[-1])

        return total_loss / num_processed
    # is dit sneller?
    
    # def train_model(self, input_data, output_data, epochs: int = 5): # without batches
    #     self.init_weigths(len(output_data[0]))
        
    #     # Train epochs
    #     for epoch in range(epochs):
    #         loss = self.train_epoch(input_data, output_data)
    #         print(f"Epoch {epoch + 1}/{epochs} - Loss: {loss:.4f}")

    #     print("Done!")

    def train_model(self, batches, epochs: int = 5): # with batches
        # keep track of time
        start_time = time.time()

        self.init_weigths(batches[0].y.shape[1])

        # Train epochs
        for epoch in range(epochs):
            start = time.time()
            loss = self.train_epoch(batches)
            print(f"Epoch {epoch + 1}/{epochs} - Loss: {loss:.4f} - Took: {round(time.time() - start, 2)}s")

        print("Done!")
        duration = time.time() - start_time
        print(f"Training took {round(duration, 2)}s")

    def test_model(self, input_data):
        return self.forward(input_data)
    
    def save_model(self, filename = "test.pkl"):
        joblib.dump(self.layers, filename, compress=3)

    def load_model(self, filename = "test.pkl"):
        self.layers = joblib.load(filename)
 