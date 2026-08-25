from neural_network.NeuralNetwork import *
from neural_network.Datasets import Wikipedia
from neural_network.Layers import Dense, Embedding
from neural_network.ActivationFunctions import *
from neural_network.LossFunctions import *
from neural_network.Optimizers import *
import tiktoken
import regex
import jax.numpy as np
import json
import joblib
import numpy

train = True # edit this if needed

wiki = Wikipedia()
text = wiki.text # [:50000]

pat_str = r"""'(?i:[sdmt]|ll|ve|re)|[^\r\n\p{L}\p{N}]?+\p{L}++|\p{N}{1,3}+| ?[^\s\p{L}\p{N}]++[\r\n]*+|\s++$|\s*[\r\n]|\s+(?!\S)|\s"""
pieces = regex.findall(pat_str, text)

should_dump = False

try:
    if train: raise FileNotFoundError("Can't load vocab when training.")
    ranks = joblib.load("data/vocab.pkl")
except FileNotFoundError:
    should_dump = True

    ranks = {bytes([i]): i for i in range(256)}
    for p in dict.fromkeys(pieces):
        b = p.encode("utf-8")
        if b not in ranks:
            ranks[b] = len(ranks)

    joblib.dump(ranks, "data/vocab.pkl")

encoding = tiktoken.Encoding("custom", pat_str=pat_str, mergeable_ranks=ranks, special_tokens={})
vocab_size = encoding.n_vocab

if should_dump:
    vocab = {encoding.decode([i]): i for i in range(encoding.n_vocab)}
    with open("data/vocab.json", "w", encoding="utf-8") as f:
        json.dump(vocab, f, ensure_ascii=False, indent=2)

if train:
    def get_skipgram_pairs(tokens: np.ndarray, window_size: int):
        centers = []
        contexts = []
        for offset in range(-window_size, window_size + 1):
            if offset == 0:
                continue
            if offset > 0:
                centers.append(tokens[:-offset])
                contexts.append(tokens[offset:])
            else:
                centers.append(tokens[-offset:])
                contexts.append(tokens[:offset])

        x_train = np.array(np.concatenate(centers))
        y_train = one_hot_encode(np.array(np.concatenate(contexts)), vocab_size)

        return x_train, y_train

    embedding_dim = 50

    tokens = np.array(encoding.encode(text)) # [:1000]

    window_size = 5

    batch_size = 64

    # source: https://mbrenndoerfer.com/writing/skip-gram-model-word2vec-word-embeddings#skip-gram-model
    X_train, Y_train = get_skipgram_pairs(tokens, window_size)
    batches = create_batches(X_train, Y_train, batch_size)

    skip_gram = NeuralNetwork(BinaryCrossEntropy(), SGD(0.05), [
        Embedding(vocab_size, embedding_dim),
        Dense(embedding_dim, Sigmoid(), False)
    ])
    skip_gram.train_model(batches, 20)
    save_model(skip_gram, "data/skipgram.pkl")

skip_gram = load_model("data/skipgram.pkl")

def get_vector(tokens: np.ndarray):
    return skip_gram.layers[0].weights[tokens]

def get_token(embedding: np.ndarray):
    # print(numpy.asarray(skip_gram.layers[0].weights == embedding).nonzero())
    # Should be [0][0] but why does one the first embedding vector have a different index sometimes?
    return numpy.where((skip_gram.layers[0].weights == embedding).all(axis=-1))[0][0]

def cosine_sim(v1: np.ndarray, v2: np.ndarray): # calculates cos(angle) of two vectors
    return (v1 @ v2) / (np.linalg.norm(v1) * np.linalg.norm(v2))

def get_similar(embedding: np.ndarray):
    similarities = cosine_sim(skip_gram.layers[0].weights, np.reshape(embedding, (50, 1)))
    indices = np.argsort(np.ravel(similarities))
    return indices

vectors = get_vector(np.array(encoding.encode(" month year April August first second"))) # must begin with leading space
# print(numpy.array2string(numpy.array(vectors), precision=3, suppress_small=True, threshold=100, edgeitems=50))
# print(cosine_sim(vectors[0], vectors[1]))
# print(cosine_sim(vectors[0], vectors[2]))
# print(cosine_sim(vectors[2], vectors[3]))

for v in vectors:
    print(encoding.decode([get_token(v)]), end="")
    print(" -> ", end="")
    print(encoding.decode([get_similar(v)[-2]]))

# month = encoding.encode("months")[0]
indices = get_similar(vectors[0])
# print(month)
# print(np.where(indices == month)[0][0])
# print(encoding.decode([np.where(indices == month)[0][0]]))
# print(encoding.decode([indices[-1]]))

# print(skip_gram.layers[0].weights.shape)