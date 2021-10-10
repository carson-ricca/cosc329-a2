import os
import re
import shutil
from pathlib import Path

import pandas as pd
from nltk.stem import PorterStemmer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from scipy.cluster.hierarchy import dendrogram, linkage
from matplotlib import pyplot as plt

from stop_words import get_stop_words

ps = PorterStemmer()

BASE_PATH = "data"
FILES_PATH = BASE_PATH + "/raw/"
PROCESSED_FILE_PATH = BASE_PATH + "/processed/"
DENDROGRAM_PATH = BASE_PATH + "/dendrogram/"
TF_IDF_OUTPUT_PATH = BASE_PATH + "/tf_idf_output/"
COSINE_SIMILARITY_PATH = BASE_PATH + "/cosine_similarity/"


def process_files(words):
    """
    This function does all the preprocessing of the txt files. It converts everything to lowercase, removes all the
    stop-words, punctuations, and non-word characters. It then applies the Porter stemming algorithm a single time. I
    also removed all words that were 3 characters or less.
    :param words: the stop words used in the process function to be removed during processing.
    :return:
    """
    if Path(PROCESSED_FILE_PATH).exists() and Path(PROCESSED_FILE_PATH).is_dir():
        shutil.rmtree(Path(PROCESSED_FILE_PATH))
    Path(PROCESSED_FILE_PATH).mkdir(parents=True, exist_ok=True)
    for file_name in os.listdir(FILES_PATH):
        if file_name.endswith(".txt"):
            with open(FILES_PATH + file_name) as file:
                contents = file.readlines()
                processed_contents = ""
                for i, line in enumerate(contents):
                    line.lower()
                    for word in words:
                        regex = r"\b" + re.escape(word) + r"\b"
                        line = re.sub(regex, "", line)
                    line = re.sub(r"[^A-Za-z0-9 ]+", " ", line)
                    line = re.sub(r"\b\w{1,3}\b", "", line)
                    processed_contents += line
                processed_word_list = []
                stemming_words = processed_contents.split()
                for word in stemming_words:
                    processed_word_list.append(ps.stem(word))
                with open(PROCESSED_FILE_PATH + "processed-" + file_name, "w") as new_file:
                    new_file.write(" ".join(processed_word_list))


def create_document_vectors():
    """
    This function creates the document vectors, and saves them to a directory.
    :return: the document vectors
    """
    if Path(TF_IDF_OUTPUT_PATH).exists() and Path(TF_IDF_OUTPUT_PATH).is_dir():
        shutil.rmtree(Path(TF_IDF_OUTPUT_PATH))
    Path(TF_IDF_OUTPUT_PATH).mkdir(parents=True, exist_ok=True)
    corpus = []
    filenames = []
    for file_name in os.listdir(PROCESSED_FILE_PATH):
        if file_name.endswith(".txt"):
            with open(PROCESSED_FILE_PATH + file_name) as file:
                corpus.append(file.read())
                filenames.append(file_name.replace(".txt", ".csv"))
    vectorizer = TfidfVectorizer()
    transformed_documents = vectorizer.fit_transform(corpus)
    transformed_documents_as_array = transformed_documents.toarray()

    for i, doc in enumerate(transformed_documents_as_array):
        tf_idf_tuples = list(zip(vectorizer.get_feature_names(), doc))
        one_doc_as_df = pd.DataFrame.from_records(tf_idf_tuples, columns=['term', 'score']) \
            .sort_values(by='score', ascending=False).reset_index(drop=True)
        one_doc_as_df.to_csv(Path(TF_IDF_OUTPUT_PATH + filenames[i]))

    return transformed_documents


def get_cosine_similarity(document_vectors):
    """
    This function creates a cosine similarity matrix for a set of document vectors, and exports it to a path.
    :param document_vectors: the document vectors to get the cosine similarity for.
    :return: the cosine similarity matrix.
    """
    if Path(COSINE_SIMILARITY_PATH).exists() and Path(COSINE_SIMILARITY_PATH).is_dir():
        shutil.rmtree(Path(COSINE_SIMILARITY_PATH))
    Path(COSINE_SIMILARITY_PATH).mkdir(parents=True, exist_ok=True)
    cos = cosine_similarity(document_vectors, document_vectors)
    df = pd.DataFrame(cos)
    df.to_csv(Path(COSINE_SIMILARITY_PATH + "matrix.csv"))
    return cos


def plot_dendrogram(model):
    """
    This function plots the dendrogram and saves it to a file.
    :param model: the model to cluster and plot.
    :return:
    """
    if Path(DENDROGRAM_PATH).exists() and Path(DENDROGRAM_PATH).is_dir():
        shutil.rmtree(Path(DENDROGRAM_PATH))
    Path(DENDROGRAM_PATH).mkdir(parents=True, exist_ok=True)
    z = linkage(model)
    dendrogram(z)
    plt.gcf()
    plt.savefig(Path(DENDROGRAM_PATH + "figure.png"))


if __name__ == "__main__":
    # Get the stop words from the csv.
    stop_words = get_stop_words()

    # Pre processing of the files.
    process_files(stop_words)

    # Create the document vectors.
    vectors = create_document_vectors()

    # Get the cosine similarity matrix for the document vectors.
    cos_similarity = get_cosine_similarity(vectors)

    # Cluster and plot dendrogram.
    plot_dendrogram(cos_similarity)
