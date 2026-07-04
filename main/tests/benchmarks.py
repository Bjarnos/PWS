# type: ignore
from neural_network.NeuralNetwork import *
from neural_network.Datasets import MNIST
from neural_network.Layers import *
from neural_network.ActivationFunctions import *
from neural_network.LossFunctions import *
from neural_network.Optimizers import *

from time import time
import json

mnist = MNIST()

def benchmark(network_loss: LossFunction, network_optimizer: Optimizer, network_layers: list[Layer]):
    # Train a new model:
    t1 = time()
    network = NeuralNetwork(loss=network_loss, optimizer=network_optimizer, layers=network_layers)
    t2 = time()
    batches = create_batches(mnist.train_images, mnist.train_labels, 32)
    t3 = time()
    network.train_model(batches)
    t4 = time()
    save_model(network, "data/test.pkl")
    t5 = time()

    # Test the model:
    network = load_model("data/test.pkl")
    t6 = time()
    test_batches = create_batches(mnist.test_images, mnist.test_labels, 16)
    t7 = time()
    final_acc = network.test_model(test_batches)
    print(f"Final accuracy: {(final_acc*100):.4f}%")
    t8 = time()
    print(
        f"Creating batches:               {t3-t2:.4}s\n"
        f"Training model:                 {t4-t3:.5}s\n"
        f"Saving model to file:           {t5-t4:.4}s\n"
        f"Loading model from file:        {t6-t5:.3}s\n"
        f"Creating second set of batches: {t7-t6:.4}s\n"
        f"Testing model:                  {t8-t7:.4}s\n"
        f"TOTAL:                          {time()-t1:.5}s"
        )
    
    return (t4-t3, final_acc)

times = []
total_runs = (12*4)*(2*7) + (12*4)*(3*8)
runs = 0
for activation in [Linear, ReLU, LeakyReLU, Softplus, ELU, SELU, GELU, Gaussian, Sigmoid, Softsign, Swish, Tanh]:
    for loss in [MeanSquaredError, MeanAbsoluteError, CategorialCrossEntropy, KLDivergence]:
        for optimizer in [SGD, SGDM, AdaGrad, RMSprop, Adam]:
            if optimizer in [Adam, RMSprop]:
                learning_rates = [0.0001, 0.0003, 0.0005, 0.001, 0.002, 0.005, 0.01]
            elif optimizer in [SGD, SGDM, AdaGrad]:
                learning_rates = [0.005, 0.01, 0.03, 0.05, 0.1, 0.2, 0.3, 0.5]

            for learning_rate in learning_rates:
                return_value = benchmark(loss(), optimizer(learning_rate), [
                    Dense(input_size=mnist.get_input_size(), activation=activation()), # input -> hidden layer
                    Dense(input_size=256, activation=Softmax()) # hidden -> output layer
                    ])
            
                times.append([activation.__name__, loss.__name__, optimizer.__name__, learning_rate, return_value[0], return_value[1]])

                runs += 1
                print(f"Completed: {runs}/{total_runs}")

with open("benchmarks.json", "w") as file:
    json.dump(times, file)
