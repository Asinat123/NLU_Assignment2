"""
CSL 7640 - Assignment 2
Problem 1: Dataset Preparation

This script:
1. Extracts text from PDF files
2. Cleans and preprocesses the text
3. Removes boilerplate content
4. Tokenizes the text
5. Computes dataset statistics
6. Generates a word cloud
7. Saves the cleaned corpus

Author: (Write your name here)
"""

import os
import re
from collections import Counter
from PyPDF2 import PdfReader
import nltk
from nltk.tokenize import word_tokenize
from wordcloud import WordCloud
import matplotlib.pyplot as plt
from nltk.corpus import stopwords
nltk.download('stopwords')

stop_words = set(stopwords.words('english'))

# Download tokenizer (only first time)
nltk.download('punkt')


# -----------------------------
# STEP 1: EXTRACT TEXT FROM PDFs
# -----------------------------
def extract_text_from_pdfs(folder_path):
    documents = []

    for file in os.listdir(folder_path):
        if file.endswith(".pdf"):
            file_path = os.path.join(folder_path, file)

            reader = PdfReader(file_path)
            text = ""

            for page in reader.pages:
                page_text = page.extract_text()
                if page_text:
                    text += page_text

            documents.append(text)

    return documents


# -----------------------------
# STEP 2: REMOVE BOILERPLATE
# -----------------------------
def remove_boilerplate(text):
    """
    Removes repeated institutional and formatting phrases
    commonly found in academic PDFs.
    """

    unwanted_phrases = [
        "indian institute of technology jodhpur",
        "iit jodhpur",
        "page",
        "course code",
        "credits",
        "semester",
    ]

    text = text.lower()

    for phrase in unwanted_phrases:
        text = text.replace(phrase, "")

    return text


# -----------------------------
# STEP 3: CLEAN TEXT
# -----------------------------
def clean_text(text):
    """
    Cleans raw extracted text by:
    - Removing numbers
    - Removing special characters
    - Normalizing whitespace
    """

    text = text.replace("\n", " ").replace("\t", " ")

    # Remove numbers
    text = re.sub(r'\d+', '', text)

    # Keep only alphabets
    text = re.sub(r'[^a-zA-Z\s]', '', text)

    # Lowercase
    text = text.lower()

    # Remove extra spaces
    text = re.sub(r'\s+', ' ', text).strip()

    return text


# -----------------------------
# STEP 4: PREPROCESS DOCUMENTS
# -----------------------------
def preprocess_documents(raw_docs):
    corpus = []

    for doc in raw_docs:
        doc = remove_boilerplate(doc)
        cleaned = clean_text(doc)

        # SIMPLE TOKENIZATION (RECOMMENDED)
        tokens = cleaned.split()

        # Remove stopwords
        tokens = [word for word in tokens if word not in stop_words]
        tokens = [word for word in tokens if len(word) > 2]

        corpus.append(tokens)

    return corpus


# -----------------------------
# STEP 5: SAVE CLEAN CORPUS
# -----------------------------
def save_corpus(corpus, output_file="clean_corpus.txt"):
    with open(output_file, "w", encoding="utf-8") as f:
        for doc in corpus:
            f.write(" ".join(doc) + "\n")


# -----------------------------
# STEP 6: DATASET STATISTICS
# -----------------------------
def compute_statistics(corpus):
    all_tokens = [word for doc in corpus for word in doc]

    num_docs = len(corpus)
    num_tokens = len(all_tokens)
    vocab_size = len(set(all_tokens))

    print("\n===== DATASET STATISTICS =====")
    print("Total Documents:", num_docs)
    print("Total Tokens:", num_tokens)
    print("Vocabulary Size:", vocab_size)

    # Top 10 frequent words
    freq = Counter(all_tokens)
    print("\nTop 10 Frequent Words:")
    for word, count in freq.most_common(10):
        print(f"{word}: {count}")

    return all_tokens


# -----------------------------
# STEP 7: WORD CLOUD
# -----------------------------
def generate_wordcloud(all_tokens):
    text = " ".join(all_tokens)

    wc = WordCloud(width=800, height=400, background_color='white').generate(text)

    plt.figure(figsize=(10, 5))
    plt.imshow(wc)
    plt.axis("off")
    plt.title("Word Cloud")
    plt.show()


# -----------------------------
# MAIN FUNCTION
# -----------------------------
def main():
    pdf_folder = "Documents/"   # 🔴 CHANGE THIS to your folder

    print("Extracting text from PDFs...")
    raw_docs = extract_text_from_pdfs(pdf_folder)

    print("Preprocessing documents...")
    corpus = preprocess_documents(raw_docs)

    print("Saving cleaned corpus...")
    save_corpus(corpus)

    print("Computing statistics...")
    all_tokens = compute_statistics(corpus)

    print("Generating word cloud...")
    generate_wordcloud(all_tokens)

    print("\n✅ Preprocessing Completed Successfully!")


# Run the script
if __name__ == "__main__":
    main()