# NLU Assignment 2: Word Embeddings & RNN Name Generation
**Tanisha Saini (B23CM1042)**

This repository contains the implementation and analysis for two core Natural Language Understanding tasks: training domain-specific word embeddings and character-level sequence generation.

---

## 📂 Project Structure

### Problem 1: Word Embeddings from IIT Jodhpur Data
Focuses on training Word2Vec models on a custom academic corpus (30,993 tokens) to capture domain-specific semantic relationships.
* **`problem1_task1.py`**: PDF text extraction and specialized preprocessing (tokenization, stopword removal).
* **`problem1_task2.py`**: Training CBOW and Skip-gram architectures with hyperparameter tuning (dimensions, window size, negative sampling).
* **`problem1_task3.py`**: Semantic evaluation via nearest neighbors and analogy experiments.
* **`problem1_task4.py`**: Dimensionality reduction and visualization using PCA and t-SNE.

### Problem 2: Character-Level Name Generation
Explores recurrent neural architectures for generating Indian names from a dataset of 1,000 samples.
* **`problem2.py`**: Implementation of three architectures from scratch:
    * **Vanilla RNN**: Baseline sequential model.
    * **BiLSTM**: Bidirectional context modeling (207,644 parameters).
    * **Attention-RNN**: Sequence modeling with a basic attention mechanism.

---

## 🚀 How to Run

### 1. Requirements
Ensure you have Python 3.8+ and the following libraries installed:
```bash
pip install torch gensim nltk pandas matplotlib scikit-learn
