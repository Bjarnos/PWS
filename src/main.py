from NeuralNetwork import *
from MNIST import *
from Layer import *
from ActivationFunction import *

mnist = MNIST()

network = NeuralNetwork([
    Dense(mnist.get_input_size(), activation=ReLU), # input -> hidden layer
    Dense(397, activation=Softmax) # hidden -> output layer
])
network.train_model(mnist.train_images, mnist.train_labels)
