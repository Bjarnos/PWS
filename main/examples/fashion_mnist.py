# type: ignore
from neural_network.NeuralNetwork import *
from neural_network.Datasets import FASHION_MNIST
from neural_network.Layers import Dense
from neural_network.ActivationFunctions import ReLU, Softmax
from neural_network.LossFunctions import MeanSquaredError
from neural_network.Optimizers import SGD

# Initialize our dataset:
mnist = FASHION_MNIST()

# Train a new model:
network = NeuralNetwork(loss=MeanSquaredError(), optimizer=SGD(), layers=[
    Dense(input_size=mnist.get_input_size(), activation=ReLU()), # input -> hidden layer
    Dense(input_size=256, activation=Softmax()) # hidden -> output layer
    ])
batches = create_batches(mnist.train_images, mnist.train_labels, 32)
network.train_model(batches, epochs=10)
save_model(network, "data/fashion-mnist.pkl")

# Test the model:
test_batches = create_batches(mnist.test_images, mnist.test_labels, 16)
print(f"Final accuracy: {(network.test_model(test_batches)*100):.4}%")
