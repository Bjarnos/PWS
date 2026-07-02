# source: https://github.com/joohei/mnist-from-scratch
from collections.abc import Sequence
import jax.numpy as np
import numpy
import joblib # pyright: ignore[reportMissingTypeStubs]
import time
import random
import platform

from .LossFunction import LossFunction
from .Layer import Layer
from .Colors import theme

class Batch:
    def __init__(self, x_train: np.ndarray, y_train: np.ndarray):
        self.x = np.asarray(x_train) # pyright: ignore[reportUnknownMemberType]
        self.y = np.asarray(y_train) # pyright: ignore[reportUnknownMemberType]

def create_batches(x: np.ndarray, y: np.ndarray, size: int = 256):
    num_batches = int(np.ceil(x.shape[0] / size))
    x_train = numpy.array_split(numpy.asarray(x), num_batches)
    y_train = numpy.array_split(numpy.asarray(y), num_batches)
    return [Batch(_x, _y) for _x, _y in zip(x_train, y_train)] # pyright: ignore[reportArgumentType]

class NeuralNetwork:
    def __init__(self, loss: LossFunction, layers: Sequence[Layer]):
        self.loss = loss
        self.layers = layers
        self.learning_rate = 0.01
        self.momentum = 0.9
        self.clip_value = 5.0

    def init_weigths(self, output_layer_size: int):
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
            current_layer.weights = np.asarray(numpy.random.randn(input_size, output_size) * np.sqrt(2.0 / input_size)) # pyright: ignore[reportUnknownMemberType]
            current_layer.biases = np.zeros(output_size) # pyright: ignore[reportUnknownMemberType]
            current_layer.prev_weight_momentum = np.zeros_like(current_layer.weights) # pyright: ignore[reportUnknownMemberType]
            current_layer.prev_bias_momentum = np.zeros_like(current_layer.biases) # pyright: ignore[reportUnknownMemberType]

    def forward(self, inputs: np.ndarray):
        for layer in self.layers:
            inputs = layer.forward(inputs)

        return inputs
    
    def train_epoch(self, batches: list[Batch]) -> float:
        total_loss = 0
        num_processed = 0

        batch_amount = len(batches)
        
        shuffled_batches = batches.copy()
        random.shuffle(shuffled_batches)

        filled_char = "█"
        empty_char = "░"

        if platform.system() == "Windows":
            filled_char = "#"
            empty_char = "_"
        
        for i, batch in enumerate(shuffled_batches, 1):
            percentage = int(i/batch_amount*100)
            chars = int(percentage/5)
            
            if i % 50 == 0 or i == batch_amount:
                print(
                    f"\r\033[K{theme.PROGRESS}Progress: {theme.RESET}["
                    f"{theme.BAR_FILLED}{chars*filled_char}{theme.RESET}{theme.BAR_EMPTY}{(20-chars)*empty_char}"
                    f"{theme.RESET}] {theme.VALUE}{percentage}%",
                    end="", flush=True
                )

            outputs = self.forward(batch.x)

            num_processed += batch.x.shape[0]
            total_loss += self.loss.calculate(predicted=outputs, expected=batch.y)

            gradient = self.loss.derivative(outputs, batch.y)

            for layer in reversed(self.layers):
                gradient = layer.backward(gradient, self.learning_rate, self.momentum, self.clip_value, layer == self.layers[-1])

        return float(total_loss / num_processed)

    def train_model(self, batches: list[Batch], epochs: int = 5):
        # keep track of time
        start_time = time.time()

        self.init_weigths(batches[0].y.shape[1])

        # Train epochs
        for epoch in range(epochs):
            start = time.time()
            loss = self.train_epoch(batches)
            acc = self.test_model(batches)
            print(
                f"\r\033[K{theme.HEADER}Epoch {epoch + 1}/{epochs}{theme.RESET} - "
                f"{theme.LABEL}Loss: {theme.RESET + theme.VALUE}{loss:.4f}{theme.RESET} - "
                f"{theme.LABEL}Acc: {theme.RESET + theme.VALUE}{acc:.4f}{theme.RESET} - "
                f"{theme.LABEL}Took: {theme.RESET + theme.VALUE}{round(time.time() - start, 2)}s",
                flush=True
                )
            
            if acc == 1:
                print(f"{theme.TIME}Traing stopped! Accuracy is {theme.RESET + theme.VALUE}100%")
                break

        duration = time.time() - start_time
        print(f"{theme.TIME}Training took {theme.RESET + theme.VALUE}{round(duration, 2)}s")

    def run_chances(self, input_data: np.ndarray):
        return self.forward(input_data)
    
    def run(self, input_data: np.ndarray):
        chances = self.run_chances(input_data)
        return np.argmax(chances, axis=-1)
    
    def test_model(self, input_data: list[Batch]):
        num_correct = 0
        num_tried = 0
        for batch in input_data:
            output_data = self.forward(batch.x)
            num_tried += len(output_data)
            
            predicted_classes = np.argmax(output_data, axis=-1)
            expected_classes = np.argmax(batch.y, axis=-1)
            num_correct += int(np.sum(predicted_classes == expected_classes))
        
        return num_correct / num_tried
    
    def interactive(self):
        try:
            from .Interactive import interactive_mode
            interactive_mode(self)
        except ImportError:
            raise RuntimeError("Interactive mode libraries not installed.")
    
def save_model(network: NeuralNetwork, filename: str = "data/model.pkl", compression_level: int = 3):
    joblib.dump(network, filename, compress=compression_level) # pyright: ignore[reportUnknownMemberType]

def load_model(filename: str = "data/model.pkl"):
    return joblib.load(filename) # pyright: ignore[reportUnknownMemberType]