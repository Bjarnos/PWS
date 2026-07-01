from NeuralNetwork import *
from MNIST import MNIST
from Layer import *
from ActivationFunction import *
from LossFunction import *

mnist = MNIST()

# Train a new model:
network = NeuralNetwork(loss=MeanAbsoluteError(), layers=[
    Dense(input_size=mnist.get_input_size(), activation=ReLU), # input -> hidden layer
    Dense(input_size=256, activation=Softmax) # hidden -> output layer
])
batches = create_batches(mnist.train_images, mnist.train_labels, 32)
network.train_model(batches)
save_model(network, "test.pkl")

# Test the model:
network = load_model("test.pkl")
test_batches = create_batches(mnist.test_images, mnist.test_labels, 16)
print(network.test_model(test_batches))