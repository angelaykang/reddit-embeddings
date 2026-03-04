import os
import re
import json
import argparse
from collections import Counter

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.metrics import (
    silhouette_score,
    silhouette_samples,
    davies_bouldin_score,
    calinski_harabasz_score,
    pairwise_distances,
)
from sklearn.decomposition import PCA
from sklearn.preprocessing import normalize

import pymysql
from gensim.models import Word2Vec

import nltk
import ssl

try:
    _create_unverified_https_context = ssl._create_unverified_context
except AttributeError:
    pass
else:
    ssl._create_default_https_context = _create_unverified_https_context

nltk.download("stopwords", quiet=True)
from nltk.corpus import stopwords

_nltk_sw = set(stopwords.words("english"))
STOPWORDS = _nltk_sw | {w.replace("'", "") for w in _nltk_sw}

# Getting necessary functions from doc2vec_embeddings.py to avoid duplication and maintain consistency
# Many of these functions were originally written in the reddit_forum_analysis lab.
from doc2vec_embeddings import (
    get_conn,
    load_posts,
    build_documents,
    tokenize_docs,
    cosine_normalize,
    find_best_k,
    run_kmeans,
    compute_metrics,
    extract_cluster_keywords,
    nearest_to_centroid,
    subreddit_distribution,
    plot_elbow_silhouettes,
    plot_clusters_2d,
    plot_cluster_sizes,
    plot_per_cluster_silhouettes,
)

# MySQL Config
MYSQL_HOST = os.environ.get("MYSQL_HOST", "localhost")
MYSQL_PORT = int(os.environ.get("MYSQL_PORT", "3306"))
MYSQL_USER = os.environ.get("MYSQL_USER", "root")
MYSQL_DATABASE_NAME = os.environ.get("MYSQL_DATABASE", "reddit_forum")
MYSQL_PASSWORD = os.environ.get("MYSQL_PASSWORD", "")

# Set seed for reproducibility
np.random.seed(42)


# Word2Vec Training Function
def train_word2vec(tokens, vector_size=100, min_count=2, epochs=30, workers=4, seed=42):
    print(
        f"  Training Word2Vec: vector_size={vector_size}, "
        f"min_count={min_count}, epochs={epochs}"
    )

    model = Word2Vec(
        sentences=tokens,
        vector_size=vector_size,
        min_count=min_count,
        epochs=epochs,
        # Using skipgram with negative sampling (often better for capturing semantic relationships in smaller datasets)
        sg=1,
        hs=0,
        negative=5,
        seed=seed,
        workers=workers,
    )
    print(f"  Word2Vec vocabulary size: {len(model.wv)}")
    return model


# Word Binning Function (Clustering into n_bins using KMeans on L2 normalized vectors)
def bin_words(w2v_model, n_bins):
    # word_to_bin: dict mapping vocab word to its assigned bin id
    # km: fitted KMeans object
    vocab_words = list(w2v_model.wv.key_to_index.keys())
    word_vectors = np.array([w2v_model.wv[w] for w in vocab_words])
    word_vectors_norm = normalize(word_vectors, norm="l2")

    print(f"Binning {len(vocab_words)} words into {n_bins} bins.")
    km = KMeans(n_clusters=n_bins, random_state=42, n_init=10)
    labels = km.fit_predict(word_vectors_norm)

    word_to_bin = {w: int(labels[i]) for i, w in enumerate(vocab_words)}
    return word_to_bin, km


# Document Vectory Building Function
def build_bow_vectors(tokens_list, word_to_bin, n_bins):
    # how many words in each bin for each doc, divided by total countable words in that doc (skipping words not in the Word2Vec vocab)
    vectors = np.zeros((len(tokens_list), n_bins), dtype=np.float64)
    skipped_docs = 0

    # iterating over tokens to count and bin
    for i, doc_tokens in enumerate(tokens_list):
        countable = [t for t in doc_tokens if t in word_to_bin]
        for t in countable:
            vectors[i, word_to_bin[t]] += 1.0
        if len(countable) > 0:
            vectors[i] /= len(countable)
        else:
            skipped_docs += 1

    # Showing how many docs had zero recognized words
    if skipped_docs:
        print(f"NOTE: {skipped_docs} documents had zero countable words.")
    return vectors
