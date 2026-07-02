# type: ignore
from neural_network import *

# Initialize our dataset:
mnist = MNIST()

# Train a new model:
network = NeuralNetwork(loss=MeanSquaredError(), layers=[
    Dense(input_size=mnist.get_input_size(), activation=ReLU()), # input -> hidden layer
    Dense(input_size=256, activation=Softmax()) # hidden -> output layer
    ])
batches = create_batches(mnist.train_images, mnist.train_labels, 32)
network.train_model(batches)
save_model(network, "data/mnist.pkl")

# Test the model:
test_batches = create_batches(mnist.test_images, mnist.test_labels, 16)
print(f"Final accuracy: {(network.test_model(test_batches)*100):.4}%")
