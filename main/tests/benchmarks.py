# type: ignore
from neural_network import *
from time import time
import json

mnist = MNIST()

def benchmark(network_loss: type[LossFunction], network_layers: list[Layer]):
    # Train a new model:
    t1 = time()
    network = NeuralNetwork(loss=network_loss(), layers=network_layers)
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
for activation in [Linear, ReLU, LeakyReLU, Softplus, ELU, SELU, GELU, Gaussian, Sigmoid, Softsign, Swish, Tanh]:
    for loss in [CategorialCrossEntropy, MeanSquaredError]:
        return_value = benchmark(loss, [
            Dense(input_size=mnist.get_input_size(), activation=activation()), # input -> hidden layer
            Dense(input_size=256) # hidden -> output layer
            ])
        
        times.append([activation.__name__, loss.__name__, return_value[0], return_value[1]])

with open("benchmarks.json", "w") as file:
    json.dump(times, file)
