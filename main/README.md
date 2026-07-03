# Neural Network

A simple library to train Neural Networks and test them. Written for our PWS.

The documentation is live at https://bjarnos.dev/PWS.

## Installing

To install the base libraries:
```sh
pip install -e .
```

To install the base libraries, and the libraries needed for interactive mode:
```sh
pip install -e .[interactive]
```

## Writing code

Everyone knows it's hard to start with a library you've never used before,
so here's an example script to get you started:
```sh
from neural_network.NeuralNetwork import *
from neural_network.Datasets import FASHION_MNIST
from neural_network.Layers import Dense
from neural_network.ActivationFunctions import ReLU, Softmax
from neural_network.LossFunctions import MeanSquaredError
from neural_network.Optimizers import SGD

# Initialize our dataset:
mnist = MNIST()

# Train a new neural network:
network = NeuralNetwork(loss=MeanSquaredError(), optimizer=SGD(), layers=[
    Dense(input_size=mnist.get_input_size(), activation=ReLU()), # input -> hidden layer
    Dense(input_size=256, activation=Softmax()) # hidden -> output layer
    ])
batches = create_batches(mnist.train_images, mnist.train_labels, 32)
network.train_model(batches)

# Test the trained model:
test_batches = create_batches(mnist.test_images, mnist.test_labels, 16)
print(f"Final accuracy: {(network.test_model(test_batches)*100):.4}%")
```

## Building the docs

We use [`pdoc`](https://pdoc.dev/) to build our docs for us. Because pdoc can already recognize all important information,
we have chosen not to use a specific format for our docstrings, but just markdown.

To install pdoc just run this in the same environment as where you installed the libraries for neural_network:
```sh
pip install pdoc
cd main
```

Then, to live update the website while editing:
```sh
python -m pdoc --docformat=markdown --math neural_network
```

Or if you just want to build the output html:
```sh
python -m pdoc --docformat=markdown --math -o ./docs neural_network
```
