"""
CSL 7640 - Assignment 2
Problem 1: Task 4 - Word Embedding Visualization

This script:
1. Loads trained Word2Vec model
2. Selects important words
3. Applies PCA / t-SNE
4. Plots 2D visualization

Author: (Write your name here)
"""
import numpy as np
from gensim.models import Word2Vec
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt


# -----------------------------
# LOAD MODEL
# -----------------------------
model = Word2Vec.load("skip_vs100_w5_n5.model")


# -----------------------------
# SELECT WORDS TO VISUALIZE
# -----------------------------
# Choose meaningful academic words
words = [
    "student", "students", "phd", "research",
    "course", "courses", "exam", "assignment",
    "program", "degree", "admission", "registration",
    "candidacy", "proposal", "presentation"
]

# Filter words present in vocabulary
words = [w for w in words if w in model.wv]

# Get vectors
vectors = np.array([model.wv[w] for w in words])


# -----------------------------
# PCA (2D)
# -----------------------------
pca = PCA(n_components=2)
pca_result = pca.fit_transform(vectors)


# -----------------------------
# t-SNE (better visualization)
# -----------------------------
tsne = TSNE(n_components=2, random_state=42, perplexity=5)
tsne_result = tsne.fit_transform(vectors)


# -----------------------------
# PLOT FUNCTION
# -----------------------------
def plot_embeddings(result, title):
    plt.figure(figsize=(10, 7))

    x = result[:, 0]
    y = result[:, 1]

    plt.scatter(x, y)

    for i, word in enumerate(words):
        plt.annotate(word, (x[i], y[i]))

    plt.title(title)
    plt.grid()
    plt.show()


# -----------------------------
# PLOT BOTH
# -----------------------------
plot_embeddings(pca_result, "PCA Visualization")
plot_embeddings(tsne_result, "t-SNE Visualization")