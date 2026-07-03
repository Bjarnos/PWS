"""
This is the main module of our library, it contains the most important
class: [`NeuralNetwork`](/neural_network/NeuralNetwork.html#NeuralNetwork).

Used source: source: https://github.com/joohei/mnist-from-scratch
"""

from collections.abc import Sequence
import jax.numpy as np
import numpy
import joblib # pyright: ignore[reportMissingTypeStubs]
import time
import random
import platform

from .LossFunctions import LossFunction
from .Optimizers import Optimizer
from .Layers import Layer
from .Colors import theme

class Batch:
    """
    A batch is multiple pieces of training data packed in one
    object

    <em>You should not use this class directly, but rather
    [`create_batches()`](/neural_network/NeuralNetwork.html#create_batches).
    It is only exported to use for typing.</em>
    """

    def __init__(self, x_train: np.ndarray, y_train: np.ndarray):
        self.x: np.ndarray = np.asarray(x_train) # pyright: ignore[reportUnknownMemberType]
        "A numpy array of multiple pieces of training data"
        self.y: np.ndarray = np.asarray(y_train) # pyright: ignore[reportUnknownMemberType]
        "A numpy array of multiple training labels"

def create_batches(x: np.ndarray, y: np.ndarray, size: int = 256) -> list[Batch]:
    """
    A function used to create batches, which you can then pass into
    `NeuralNetwork.train_model()`
    """

    num_batches = int(np.ceil(x.shape[0] / size))
    x_train = numpy.array_split(numpy.asarray(x), num_batches)
    y_train = numpy.array_split(numpy.asarray(y), num_batches)
    return [Batch(_x, _y) for _x, _y in zip(x_train, y_train)] # pyright: ignore[reportArgumentType]

class NeuralNetwork:
    """
    The most important class in this library. It holds all information
    required to run or train a neural network. Including: the loss function,
    the optimizer and the layers.
    """

    def __init__(self, loss: LossFunction, optimizer: Optimizer, layers: Sequence[Layer]):
        self.loss: LossFunction = loss
        "A reference to the set loss function"
        self.optimizer: Optimizer = optimizer
        "A reference to the set optimizer"
        self.layers: Sequence[Layer] = layers
        "A reference to the set layers"
        self.clip_value: float = 5.0
        "The maximum value any training gradient can reach"

    def init_weights(self, output_layer_size: int):
        """
        When you create a network, the layers will still have default
        weights and biases. Use this function to initialize all the
        weights and biases of the model.

        <em>`train_model()` already calls this function for you, you will
        only need it if you don't want to train the network (because you
        want it to return random values)</em>
        """
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
            current_layer.weight_momentum = np.zeros_like(current_layer.weights) # pyright: ignore[reportUnknownMemberType]
            current_layer.bias_momentum = np.zeros_like(current_layer.biases) # pyright: ignore[reportUnknownMemberType]
            current_layer.weight_varience = np.zeros_like(current_layer.weights) # pyright: ignore[reportUnknownMemberType]
            current_layer.bias_varience = np.zeros_like(current_layer.biases) # pyright: ignore[reportUnknownMemberType]
            current_layer.acc_w_grad = np.zeros_like(current_layer.weights) # pyright: ignore[reportUnknownMemberType]
            current_layer.acc_b_grad = np.zeros_like(current_layer.biases) # pyright: ignore[reportUnknownMemberType]
    
    def run(self, inputs: np.ndarray) -> np.ndarray:
        """
        You can use this function to get an output
        from the network based on your input.

        Don't forget to call either `train_model()` or
        `init_weights()` before using this function!
        """
        
        for layer in self.layers:
            inputs = layer.forward(inputs)

        return inputs
    
    def train_epoch(self, batches: list[Batch]) -> float:
        """
        A function used to perform a single training session
        (each piece of training data is used once), which
        returns the loss after the training
        """
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

            outputs = self.run(batch.x)

            num_processed += batch.x.shape[0]
            total_loss += self.loss.calculate(predicted=outputs, expected=batch.y)

            gradient = self.loss.derivative(outputs, batch.y)

            for layer in reversed(self.layers):
                gradient = layer.backward(gradient, self.optimizer, self.clip_value, layer == self.layers[-1])

        return float(total_loss / num_processed)

    def train_model(self, batches: list[Batch], epochs: int = 5):
        """
        A function used to train the neural network on
        batches of train data. This will make it a
        specialized model
        """
        # keep track of time
        start_time = time.time()

        self.init_weights(batches[0].y.shape[1])

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
    
    def test_model(self, input_data: list[Batch]) -> float:
        """
        Get the accuracy of the current model on the input
        batches of test data
        """
        num_correct = 0
        num_tried = 0
        for batch in input_data:
            output_data = self.run(batch.x)
            num_tried += len(output_data)
            
            predicted_classes = np.argmax(output_data, axis=-1)
            expected_classes = np.argmax(batch.y, axis=-1)
            num_correct += int(np.sum(predicted_classes == expected_classes))
        
        return num_correct / num_tried
    
    def interactive(self):
        """
        A special mode meant for MNIST trained models only,
        where you can draw input data to see how the model
        classifies it
        """
        try:
            from .Interactive import interactive_mode
            interactive_mode(self)
        except ImportError:
            raise RuntimeError("Interactive mode libraries not installed.")
    
def save_model(network: NeuralNetwork, filename: str = "data/model.pkl", compression_level: int = 3):
    """
    A function to dump a trained model to a `.pkl` file
    """
    joblib.dump(network, filename, compress=compression_level) # pyright: ignore[reportUnknownMemberType]

def load_model(filename: str = "data/model.pkl") -> NeuralNetwork:
    """
    A function to load a neural network from a `.pkl` file
    """
    return joblib.load(filename) # pyright: ignore[reportUnknownMemberType]

__all__ = ["Batch", "create_batches", "NeuralNetwork", "save_model", "load_model"]
