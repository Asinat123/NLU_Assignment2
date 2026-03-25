"""
CSL 7640 - Assignment 2
Problem 1: Task 2 - Word2Vec Training with Experiments

This script:
1. Loads cleaned corpus
2. Trains CBOW and Skip-gram models
3. Performs controlled experiments:
   - Embedding dimension
   - Context window size
   - Negative sampling
4. Saves all trained models

Author: (Write your name here)
"""

from gensim.models import Word2Vec


# -----------------------------
# LOAD CORPUS
# -----------------------------
def load_corpus(file_path):
    """
    Loads tokenized corpus from file.
    Each line represents one document.
    """
    corpus = []

    with open(file_path, "r", encoding="utf-8") as f:
        for line in f:
            tokens = line.strip().split()
            corpus.append(tokens)

    return corpus


# -----------------------------
# TRAIN MODEL FUNCTION
# -----------------------------
def train_model(corpus, vector_size, window, negative, sg):
    """
    Trains a Word2Vec model.

    Parameters:
    - vector_size: embedding dimension
    - window: context window size
    - negative: number of negative samples
    - sg: 0 = CBOW, 1 = Skip-gram
    """

    model = Word2Vec(
        sentences=corpus,
        vector_size=vector_size,
        window=window,
        min_count=2,
        sg=sg,
        negative=negative,
        workers=4
    )

    return model


# -----------------------------
# RUN CONTROLLED EXPERIMENTS
# -----------------------------
def run_experiments(corpus):
    """
    Runs experiments by varying one parameter at a time.
    """

    experiments = []

    # -------------------------
    # 1. EMBEDDING SIZE EXPERIMENT
    # -------------------------
    print("\n=== Embedding Size Experiment ===")
    for size in [50, 100, 200]:
        config = {"vector_size": size, "window": 5, "negative": 5}
        experiments.append(config)

    # -------------------------
    # 2. WINDOW SIZE EXPERIMENT
    # -------------------------
    print("\n=== Window Size Experiment ===")
    for window in [3, 5, 8]:
        config = {"vector_size": 100, "window": window, "negative": 5}
        experiments.append(config)

    # -------------------------
    # 3. NEGATIVE SAMPLING EXPERIMENT
    # -------------------------
    print("\n=== Negative Sampling Experiment ===")
    for neg in [3, 5, 10]:
        config = {"vector_size": 100, "window": 5, "negative": neg}
        experiments.append(config)

    # Remove duplicates (since 100,5,5 appears multiple times)
    unique_experiments = [dict(t) for t in {tuple(d.items()) for d in experiments}]

    results = []

    # -------------------------
    # TRAIN MODELS
    # -------------------------
    for config in unique_experiments:
        print("\n------------------------------")
        print("Training Config:", config)

        # CBOW model (sg=0)
        cbow_model = train_model(corpus, **config, sg=0)
        cbow_name = f"cbow_vs{config['vector_size']}_w{config['window']}_n{config['negative']}.model"
        cbow_model.save(cbow_name)

        # Skip-gram model (sg=1)
        skip_model = train_model(corpus, **config, sg=1)
        skip_name = f"skip_vs{config['vector_size']}_w{config['window']}_n{config['negative']}.model"
        skip_model.save(skip_name)

        print("Saved:", cbow_name, "and", skip_name)

        results.append((config, cbow_model, skip_model))

    return results


# -----------------------------
# MAIN FUNCTION
# -----------------------------
def main():
    corpus_file = "clean_corpus.txt"

    print("Loading corpus...")
    corpus = load_corpus(corpus_file)

    print("Running experiments...")
    results = run_experiments(corpus)

    print("\n✅ All experiments completed successfully!")


# -----------------------------
# RUN SCRIPT
# -----------------------------
if __name__ == "__main__":
    main()