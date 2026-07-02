# type: ignore
import numpy as np
import matplotlib.pyplot as plt
from neural_network import *

# Create a network:
# input: xyr (radius), output: rgb (sigmoid returns 0-1)
network = NeuralNetwork(loss=MeanSquaredError(), layers=[
    Dense(input_size=3, activation=Tanh()),
    Dense(input_size=32, activation=Tanh()),
    Dense(input_size=32, activation=Tanh()),
    Dense(input_size=32, activation=Tanh()),
    Dense(input_size=32, activation=Sigmoid())
])

# The model now has random weights, let's not train it :P
network.init_weigths(3)

# Draw some random RGB art:
size = 256
x = np.linspace(-1, 1, size)
y = np.linspace(-1, 1, size)
X, Y = np.meshgrid(x, y)

# Add a radius feature so it looks angular
R = np.sqrt(X**2 + Y**2)

inputs = np.stack([X.flatten(), Y.flatten(), R.flatten()], axis=-1)

colors = network.run(inputs)
image = np.array(colors).reshape((size, size, 3))

plt.figure(figsize=(8, 8))
plt.imshow(image)
plt.axis('off')
plt.title("Beautiful Angular Art")
plt.tight_layout()
plt.show()
