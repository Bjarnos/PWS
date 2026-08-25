from neural_network.NeuralNetwork import *
from neural_network.Datasets import Wikipedia
from neural_network.Layers import Dense
from neural_network.ActivationFunctions import *
from neural_network.LossFunctions import *
from neural_network.Optimizers import *
import tiktoken
import jax.numpy as np
import numpy

encoding = tiktoken.get_encoding("cl100k_base")
vocab_size = encoding.n_vocab

train = False

if train:
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

    embedding_dim = 50

    tokens = np.array(encoding.encode(wiki.text[:10000]))[:1000]

    window_size = 5

    batch_size = 32

    # source: https://mbrenndoerfer.com/writing/skip-gram-model-word2vec-word-embeddings#skip-gram-model
    X_train, Y_train = get_skipgram_pairs(tokens, window_size)
    batches = create_batches(X_train, Y_train, batch_size)

    skip_gram = NeuralNetwork(CategorialCrossEntropy(), SGD(), [
        Dense(vocab_size, Linear(), False),
        Dense(embedding_dim, Softmax(), False)
    ])
    skip_gram.train_model(batches, 20)
    save_model(skip_gram, "data/skipgram.pkl")

skip_gram = load_model("data/skipgram.pkl")


wiki = Wikipedia()
print(wiki.text[:10000])
def get_vector(tokens: np.ndarray):
    return skip_gram.layers[0].weights[tokens]

def get_token(embedding: np.ndarray):
    # print(numpy.asarray(skip_gram.layers[0].weights == embedding).nonzero())
    # Should be [0][0] but why does one the first embedding vector have a different index sometimes?
    return numpy.where(skip_gram.layers[0].weights == embedding)[0][1]

def cosine_sim(v1: np.ndarray, v2: np.ndarray): # calculates cos(angle) of two vectors
    return (v1 @ v2) / (np.linalg.norm(v1) * np.linalg.norm(v2))

def get_similar(embedding: np.ndarray):
    similarities = cosine_sim(skip_gram.layers[0].weights, np.reshape(embedding, (50, 1)))
    indices = np.argsort(np.ravel(similarities))
    return indices

vectors = get_vector(np.array(encoding.encode("month year april august pizza first, second")))
# print(numpy.array2string(numpy.array(vectors), precision=3, suppress_small=True, threshold=100, edgeitems=50))
# print(cosine_sim(vectors[0], vectors[1]))
# print(cosine_sim(vectors[0], vectors[2]))
# print(cosine_sim(vectors[2], vectors[3]))

for v in vectors:
    print(encoding.decode([get_token(v)]), end="")
    print(" -> ", end="")
    print(encoding.decode([get_similar(v)[-1]]))

# month = encoding.encode("months")[0]
indices = get_similar(vectors[0])
# print(month)
# print(np.where(indices == month)[0][0])
# print(encoding.decode([np.where(indices == month)[0][0]]))
# print(encoding.decode([indices[-1]]))

# print(skip_gram.layers[0].weights.shape)