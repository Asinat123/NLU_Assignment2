"""
CSL 7640 - Assignment 2
Problem 1: Task 3 - Semantic Analysis (Final Version)

This script:
1. Loads all trained Word2Vec models
2. Computes nearest neighbors for selected words
3. Performs analogy experiments based on dataset vocabulary

Author: (Write your name here)
"""

from gensim.models import Word2Vec
import os


# -----------------------------
# LOAD MODEL
# -----------------------------
def load_model(model_path):
    return Word2Vec.load(model_path)


# -----------------------------
# NEAREST NEIGHBORS
# -----------------------------
def nearest_neighbors(model, words):
    """
    Finds top 5 similar words using cosine similarity.
    """
    for word in words:
        print(f"\nWord: {word}")

        if word in model.wv:
            neighbors = model.wv.most_similar(word, topn=5)
            for w, score in neighbors:
                print(f"{w} ({score:.3f})")
        else:
            print("Word not in vocabulary")


# -----------------------------
# ANALOGY FUNCTION
# -----------------------------
def analogy(model, positive, negative):
    """
    Performs analogy: positive - negative
    """
    try:
        result = model.wv.most_similar(
            positive=positive,
            negative=negative,
            topn=3
        )

        for w, score in result:
            print(f"{w} ({score:.3f})")

    except KeyError:
        print("Words missing in vocabulary")


# -----------------------------
# MAIN FUNCTION
# -----------------------------
def main():
    # Get all model files in current directory
    model_files = [f for f in os.listdir() if f.endswith(".model")]

    # Words required by assignment
    words = ["research", "student", "phd", "exam"]

    for model_file in model_files:
        print("\n======================================")
        print("MODEL:", model_file)
        print("======================================")

        model = load_model(model_file)

        # -------------------------
        # NEAREST NEIGHBORS
        # -------------------------
        print("\n--- Nearest Neighbors ---")
        nearest_neighbors(model, words)

        # -------------------------
        # ANALOGY TASKS (UPDATED)
        # -------------------------
        print("\n--- Analogy Tasks ---")

        print("\nAnalogy 1: PhD : Student :: Course : ?")
        analogy(model,
                positive=["phd", "student"],
                negative=["course"])

        print("\nAnalogy 2: Exam : Student :: Assignment : ?")
        analogy(model,
                positive=["exam", "student"],
                negative=["course"])

        print("\nAnalogy 3: PhD : Research :: Course : ?")
        analogy(model,
                positive=["phd", "research"],
                negative=["course"])


# -----------------------------
# RUN SCRIPT
# -----------------------------
if __name__ == "__main__":
    main()