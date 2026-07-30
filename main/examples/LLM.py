from neural_network.NeuralNetwork import *
from neural_network.Datasets import Wikipedia
from neural_network.Layers import Dense
from neural_network.ActivationFunctions import *
from neural_network.LossFunctions import *
from neural_network.Optimizers import *
import tiktoken
import jax.numpy as np
import numpy
import time

def get_skipgram_pairs(tokens: np.ndarray, window_size: int):
    x_train = []
    y_train = []
    for center_idx in range(tokens.size):
        center = tokens[center_idx]
        for offset in range(-window_size, window_size + 1):
            context_idx = center_idx + offset

            if (offset == 0 or 0 < context_idx >= tokens.size):
                continue

            context = tokens[context_idx]
            x_train.append(center)
            y_train.append(context)

    x_train = one_hot_encode(np.array(x_train), vocab_size)
    y_train = one_hot_encode(np.array(y_train), vocab_size)

    return x_train, y_train


wiki = Wikipedia()
#print(wiki.text[:50])

encoding = tiktoken.get_encoding("cl100k_base")
vocab_size = encoding.n_vocab
embedding_dim = 50

tokens = np.array(encoding.encode(wiki.text[:1000]))

window_size = 5

batch_size = 32

# source: https://mbrenndoerfer.com/writing/skip-gram-model-word2vec-word-embeddings#skip-gram-model
X_train, Y_train = get_skipgram_pairs(tokens, window_size)
batches = create_batches(X_train, Y_train, batch_size)

skip_gram = NeuralNetwork(CategorialCrossEntropy(), Adam(), [
    Dense(vocab_size, Linear(), False),
    Dense(embedding_dim, Softmax(), False)
])
skip_gram.train_model(batches, 20)
save_model(skip_gram, "data/skipgram.pkl")


skip_gram = load_model("data/skipgram.pkl")

def get_vector(tokens: np.ndarray):
    return skip_gram.layers[0].weights[tokens]

def cosine_sim(v1: np.ndarray, v2: np.ndarray): # calculates cos(angle) of two vectors
    return float(np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2)))

vectors = get_vector(np.array(encoding.encode("month year april august")))
print(numpy.array2string(numpy.array(vectors), precision=3, suppress_small=True, threshold=100, edgeitems=50))
print(cosine_sim(vectors[0], vectors[1]))
print(cosine_sim(vectors[0], vectors[2]))
print(cosine_sim(vectors[2], vectors[3]))

# batches = create_batches()
# SkipGram.train_model(batches, 5)
print(skip_gram.layers[0].weights.shape)