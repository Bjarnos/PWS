# Neural Network

A simple library to train Neural Networks and test them. Written for our PWS.

To install the base libraries:
```sh
pip install -e .
```

To install the base libraries, and the libraries needed for interactive mode:
```sh
pip install -e .[interactive]
```

Example script, to get you started:
```sh
from neural_network import *

# Initialize our dataset:
mnist = MNIST()

# Train a new neural network:
network = NeuralNetwork(loss=MeanSquaredError(), layers=[
    Dense(input_size=mnist.get_input_size(), activation=ReLU()), # input -> hidden layer
    Dense(input_size=256, activation=Softmax()) # hidden -> output layer
    # The last activation function must be Softmax to map the vectors!
    ])
batches = create_batches(mnist.train_images, mnist.train_labels, 32)
network.train_model(batches)

# Test the trained model:
test_batches = create_batches(mnist.test_images, mnist.test_labels, 16)
print(f"Final accuracy: {(network.test_model(test_batches)*100):.4}%")
```
