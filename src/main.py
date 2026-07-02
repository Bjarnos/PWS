from NeuralNetwork import *
from MNIST import MNIST
from Layer import *
from ActivationFunction import *
from LossFunction import *

t0 = time.time()
mnist = MNIST()

# Train a new model:
t1 = time.time()
network = NeuralNetwork(loss=MeanSquaredError(), layers=[
    Dense(input_size=mnist.get_input_size(), activation=ReLU), # input -> hidden layer
    Dense(input_size=256, activation=Softmax) # hidden -> output layer
])
t2 = time.time()
batches = create_batches(mnist.train_images, mnist.train_labels, 32)
t3 = time.time()
network.train_model(batches)
t4 = time.time()
save_model(network, "test.pkl")
t5 = time.time()

# Test the model:
network = load_model("test.pkl")
t6 = time.time()
test_batches = create_batches(mnist.test_images, mnist.test_labels, 16)
t7 = time.time()
print(network.test_model(test_batches))
print(f"0: {t1-t0}s, 1: {t2 - t1}s, 2: {t3-t2}s, 3: {t4-t3}s, 4: {t5-t4}s, 5: {t6-t5}s, 6: {t7-t6}s")