# type: ignore
from neural_network.NeuralNetwork import *

network = load_model("data/test.pkl")
network.interactive()
