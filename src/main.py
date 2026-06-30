from NeuralNetwork import *
from MNIST import *
from Layer import *
from ActivationFunction import *

mnist = MNIST()

# output_size is always calculated from either the next layer's input_size or the given output_data in train_model
# this means the second layer here has an input_size of 397 but the amount of neurons is 10, confusing xD
network = NeuralNetwork(loss=CategorialCrossEntropy(1e-9), layers=[
    Dense(input_size=mnist.get_input_size(), activation=ReLU), # input -> hidden layer
    Dense(input_size=256, activation=Softmax) # hidden -> output layer
])
#network.train_model(mnist.train_images, mnist.train_labels)
batches = create_batches(mnist.train_images, mnist.train_labels, 16)
network.train_model(batches)

network.save_model()