from neural_network.NeuralNetwork import *
from neural_network.Datasets import Wikipedia
from neural_network.Layers import Dense, Embedding
from neural_network.ActivationFunctions import *
from neural_network.LossFunctions import *
from neural_network.Optimizers import *
from collections import Counter
import re
import json
import joblib
import jax.numpy as np

train = True # edit this if needed

wiki = Wikipedia()
text = wiki.text[:50000]

min_count = 5
max_vocab_size = 4000
unk_token = "_"

if train:
    words = re.findall(r"\b[a-zA-Z]+(?:'[a-zA-Z]+)?\b", text.lower()) # only allow letters
    counts = Counter(words)

    vocab = {unk_token: 0}
    for word, count in counts.most_common(max_vocab_size - 1):
        if count >= min_count and word != unk_token:
            vocab[word] = len(vocab)

    with open("data/vocab.json", "w", encoding="utf-8") as f:
        json.dump(vocab, f, ensure_ascii=False, indent=2)
    joblib.dump(vocab, "data/vocab.pkl")
else:
    with open("data/vocab.json", "r", encoding="utf-8") as f:
        vocab = json.load(f)

idx2word = list(vocab.keys())
vocab_size = len(vocab)

def encode(text: str) -> list[int]:
    words = re.findall(r"\b[a-zA-Z]+(?:'[a-zA-Z]+)?\b", text.lower())
    unk_id = vocab[unk_token]
    return [vocab.get(w, unk_id) for w in words]

def decode(tokens: list[int] | np.ndarray) -> list[str]:
    return [idx2word[i] if 0 <= i < len(idx2word) else unk_token for i in tokens]

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

    tokens = np.array(encode(text))

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

def cosine_sim(v1: np.ndarray, v2: np.ndarray):
    return (v1 @ v2) / (np.linalg.norm(v1) * np.linalg.norm(v2))

def get_similar(embedding: np.ndarray):
    similarities = cosine_sim(skip_gram.layers[0].weights, np.reshape(embedding, (50, 1)))
    indices = np.argsort(np.ravel(similarities))
    return indices

test_words = ["month", "year", "april", "august", "first", "second"]
token_ids = [vocab.get(w, vocab[unk_token]) for w in test_words]
vectors = get_vector(np.array(token_ids))

for word, token_id, v in zip(test_words, token_ids, vectors):
    similar_indices = get_similar(v)
    # the most similar is the word itself, second most similar is -2
    similar_word = idx2word[similar_indices[-2]] if len(similar_indices) > 1 else unk_token
    print(f"{word} ({token_id}) -> {similar_word}")