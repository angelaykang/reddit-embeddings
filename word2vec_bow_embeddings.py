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
def train_word2vec(tokens, vector_size=150, min_count=2, epochs=30, workers=4, seed=42):
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


# Plot BOW Config Comparison Function (slightly different than doc2vec)
def plot_config_comparison(results, outdir):
    names = [r["config"]["name"] for r in results]
    sils = [r["metrics"]["cosine_silhouette"] for r in results]
    dbs = [r["metrics"]["davies_bouldin"] for r in results]
    chs = [r["metrics"]["calinski_harabasz"] for r in results]
    stds = [r["metrics"]["cluster_size_std"] for r in results]
    colors = ["#4C72B0", "#DD8452", "#55A868"]

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    ax = axes[0, 0]
    ax.bar(names, sils, color=colors, edgecolor="white")
    ax.set_ylabel("Cosine Silhouette (higher = better)")
    ax.set_title("Cosine Silhouette Score")
    ax.grid(axis="y", alpha=0.3)
    for i, v in enumerate(sils):
        ax.text(i, v + 0.003, f"{v:.4f}", ha="center", fontsize=9)

    ax = axes[0, 1]
    ax.bar(names, dbs, color=colors, edgecolor="white")
    ax.set_ylabel("Davies-Bouldin Index (lower = better)")
    ax.set_title("Davies-Bouldin Index")
    ax.grid(axis="y", alpha=0.3)
    # 
    for i, v in enumerate(dbs):
        ax.text(i, v + 0.02, f"{v:.4f}", ha="center", fontsize=9)

    ax = axes[1, 0]
    ax.bar(names, chs, color=colors, edgecolor="white")
    ax.set_ylabel("Calinski-Harabasz (higher = better)")
    ax.set_title("Calinski-Harabasz Index")
    ax.grid(axis="y", alpha=0.3)
    for i, v in enumerate(chs):
        ax.text(i, v + max(chs) * 0.01, f"{v:.1f}", ha="center", fontsize=9)

    ax = axes[1, 1]
    ax.bar(names, stds, color=colors, edgecolor="white")
    ax.set_ylabel("Cluster-Size Std Dev (lower = more balanced)")
    ax.set_title("Cluster Balance")
    ax.grid(axis="y", alpha=0.3)
    for i, v in enumerate(stds):
        ax.text(i, v + max(stds) * 0.01, f"{v:.1f}", ha="center", fontsize=9)

    plt.suptitle("Word2Vec BoW - Cross-Configuration Comparison",
                 fontsize=14, fontweight="bold")
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    path = os.path.join(outdir, "config_comparison.png")
    plt.savefig(path, dpi=150)
    plt.close()
    print(f"\nSaved comparison plot -> {path}")


# Run Configuration Function
# Binning w2v word embeddings to n_bins clusters, building normalized word-bin frequency vectors for each doc
# Clustering doc vecs, evaluating, and plotting
def run_configuration(df, tokens, w2v_model, config, outdir, k=None, k_min=2, k_max=15):
    name = config["name"]
    n_bins = config["n_bins"]

    # Binning words to n_bins clusters
    word_to_bin, km_words = bin_words(w2v_model, n_bins)

    # Building normalized word-bin frequency vectors for each doc
    raw_embeddings = build_bow_vectors(tokens, word_to_bin, n_bins)
    print(f"Bag-of-Words matrix shape: {raw_embeddings.shape}")

    # Drop docs with no recognizable words
    row_sums = raw_embeddings.sum(axis=1)
    nonzero_mask = row_sums > 0
    if not nonzero_mask.all():
        n_dropped = int((~nonzero_mask).sum())
        print(f"Dropping {n_dropped} zero-vector documents before clustering.")
        raw_embeddings = raw_embeddings[nonzero_mask]
        df_local = df[nonzero_mask].reset_index(drop=True)
        tokens_local = [t for t, keep in zip(tokens, nonzero_mask) if keep]
    else:
        df_local = df
        tokens_local = tokens

    # Normalizing embeddings after dropping empty docs
    embeddings_norm = cosine_normalize(raw_embeddings)
    print(f"Normalised embedding matrix shape: {embeddings_norm.shape}")

    # KMeans search
    if k is None:
        best_k, k_values, inertias, silhouettes = find_best_k(embeddings_norm, k_min=k_min, k_max=k_max)
    # Fallback for declared k
    else:
        best_k = k
        k_values, inertias, silhouettes = None, None, None

    labels, km_docs = run_kmeans(embeddings_norm, best_k)

    # Evaluating metrics
    metrics = compute_metrics(embeddings_norm, labels, km_docs)

    kw_map = extract_cluster_keywords(df_local, labels, best_k, top_n=20)
    nearest = nearest_to_centroid(embeddings_norm, labels, km_docs, n=10)
    sub_dist = subreddit_distribution(df_local, labels, best_k)

    # Plotting
    tag = (f"{name} (bins={n_bins}, w2v_dim={config['w2v_vector_size']}, "
           f"w2v_mc={config['w2v_min_count']})")

    # Modified/following same format as in doc2vec_embeddings
    if k_values is not None:
        plot_elbow_silhouettes(
            k_values, inertias, silhouettes, best_k,
            os.path.join(outdir, f"elbow_sil_{name}.png"),
        )
    plot_clusters_2d(
        embeddings_norm, labels, best_k, kw_map, tag,
        os.path.join(outdir, f"clusters_2d_{name}.png"),
    )
    plot_cluster_sizes(
        labels, best_k, kw_map, tag,
        os.path.join(outdir, f"cluster_sizes_{name}.png"),
    )
    plot_per_cluster_silhouettes(
        metrics, best_k, tag,
        os.path.join(outdir, f"per_cluster_sil_{name}.png"),
    )

    # Printing metrics
    print(f"\n{'-' * 70}")
    print(f"  DETAILED CLUSTER EXAMINATION — '{name}'")
    print(f"  Overall cosine silhouette = {metrics['cosine_silhouette']:.4f}")
    print(f"  Davies-Bouldin = {metrics['davies_bouldin']:.4f}  |  "
          f"Calinski-Harabasz = {metrics['calinski_harabasz']:.1f}")
    print(f"{'-' * 70}")

    # Iterating over clusters for cluster analysis
    for cid in range(best_k):
        n_posts = metrics["cluster_sizes"][cid]
        cid_sil = metrics["per_cluster_silhouette"][cid]
        kw_str = ", ".join(f"{w}({c})" for w, c in kw_map.get(cid, []))
        subs = sub_dist[cid]
        total_in_cluster = sum(subs.values())
        sub_str = ", ".join(
            f"{s}: {c} ({c/total_in_cluster*100:.0f}%)"
            for s, c in sorted(subs.items(), key=lambda x: -x[1])
        )

        print(f"\n  Cluster {cid}  ({n_posts} posts, silhouette={cid_sil:.4f})")
        print(f"    Keywords: {kw_str}")
        print(f"    Subreddits: {sub_str}")
        print("    Representative posts (nearest to centroid, cosine distance):")

        # Iterating over centroid-nearest posts and printing
        for rank, idx in enumerate(nearest.get(cid, []), 1):
            row = df_local.iloc[idx]
            title = row["title"]
            selftext = row["selftext"]
            subreddit = row["subreddit"]
            print(f"      {rank}. [r/{subreddit}] {title}")
            if selftext.strip():
                print(f"         {selftext}")
            print("")
        print(f"  {'-' * 68}")

    # Creating JSON result
    cluster_summaries = []
    for cid in range(best_k):
        example_posts = []
        for idx in nearest.get(cid, []):
            row = df_local.iloc[idx]
            example_posts.append({
                "reddit_id": row["reddit_id"],
                "subreddit": row["subreddit"],
                "title": row["title"],
                "selftext": row["selftext"],
            })

        cluster_summaries.append({
            "cluster_id": int(cid),
            "size": metrics["cluster_sizes"][cid],
            "mean_cosine_silhouette": metrics["per_cluster_silhouette"][cid],
            "top_keywords": [{"word": w, "count": c} for w, c in kw_map.get(cid, [])],
            "subreddit_distribution": sub_dist[cid],
            "representative_posts": example_posts,
        })

    return {
        "config": config,
        "k": best_k,
        "metrics": {
            "cosine_silhouette": metrics["cosine_silhouette"],
            "per_cluster_silhouette": {
                str(k_): v for k_, v in metrics["per_cluster_silhouette"].items()
            },
            "davies_bouldin": metrics["davies_bouldin"],
            "calinski_harabasz": metrics["calinski_harabasz"],
            "cluster_size_std": metrics["cluster_size_std"],
            "cluster_sizes": metrics["cluster_sizes"],
        },
        "k_search": {
            "k_values": k_values,
            "inertias": (
                [float(x) for x in inertias] if inertias else None
            ),
            "silhouettes": (
                [float(x) if not np.isnan(x) else None for x in silhouettes]
                if silhouettes else None
            ),
        },
        "cluster_summaries": cluster_summaries,
    }


# Main Function
def main():
    parser = argparse.ArgumentParser(
        description="Lab 8 - Word2Vec Bag-of-Words embeddings & clustering for Reddit posts"
    )
    parser.add_argument(
        "--k", type=int, default=None,
        help="Fixed k for all configs; omit to auto-select via cosine silhouette.",
    )
    parser.add_argument("--min-k", type=int, default=2,
                        help="Minimum k for auto-selection.")
    parser.add_argument("--max-k", type=int, default=15,
                        help="Maximum k for auto-selection.")
    parser.add_argument(
        "--outdir", type=str, default="w2v_bow_results",
        help="Directory for plots and JSON summaries.",
    )
    # Extra CLI args for word vector dims for binning, min word frequency, and w2v training epochs
    parser.add_argument(
        "--w2v-dim", type=int, default=150,
        help="Word2Vec vector dimensionality for word binning (default=150).",
    )
    parser.add_argument(
        "--w2v-min-count", type=int, default=2,
        help="Word2Vec minimum word frequency (default=2).",
    )
    parser.add_argument(
        "--w2v-epochs", type=int, default=30,
        help="Word2Vec training epochs (default=30).",
    )
    args = parser.parse_args()
    os.makedirs(args.outdir, exist_ok=True)

    # Load Data
    conn = get_conn()
    try:
        df = load_posts(conn)
    finally:
        conn.close()

    if df.empty:
        print("No posts found in database. Run reddit_forum_analysis.py first.")
        return
    
    # Get docs and tokenize
    docs = build_documents(df)
    tokens = tokenize_docs(docs)
    non_empty_idx = [i for i, toks in enumerate(tokens) if toks]
    if not non_empty_idx:
        print("All posts are empty after preprocessing; nothing to embed.")
        return
    if len(non_empty_idx) < len(df):
        dropped = len(df) - len(non_empty_idx)
        print(f"Dropping {dropped} empty posts before Word2Vec training.")
        df = df.iloc[non_empty_idx].reset_index(drop=True)
        docs = [docs[i] for i in non_empty_idx]
        tokens = [tokens[i] for i in non_empty_idx]

    print(f"Built {len(docs)} non-empty documents, vocabulary tokens ready.\n")

    # Training Word2Vec (reused across bin count configs)
    word2vec_model = train_word2vec(
        tokens,
        vector_size=args.w2v_dim,
        min_count=args.w2v_min_count,
        epochs=args.w2v_epochs,
    )

    # Doc2Vec equivalent vector size configurations
    # Using CLI input vector dim and word frequency min count
    configs = [
        {
            "name": "small",
            "n_bins": 50,
            "w2v_vector_size": args.w2v_dim,
            "w2v_min_count": args.w2v_min_count,
        },
        {
            "name": "medium",
            "n_bins": 100,
            "w2v_vector_size": args.w2v_dim,
            "w2v_min_count": args.w2v_min_count,
        },
        {
            "name": "large",
            "n_bins": 200,
            "w2v_vector_size": args.w2v_dim,
            "w2v_min_count": args.w2v_min_count,
        },
    ]

    all_results = []
    best_result = None
    best_silhouette = float("-inf")

    for cfg in configs:
        print("\n" + "=" * 80)
        print(f"  CONFIGURATION: {cfg['name'].upper()}")
        print(f"    n_bins (document vector dim) = {cfg['n_bins']}")
        print(f"    Word2Vec dim                 = {cfg['w2v_vector_size']}")
        print(f"    Word2Vec min_count           = {cfg['w2v_min_count']}")
        print("=" * 80)

        result = run_configuration(
            df, tokens, word2vec_model, cfg, args.outdir,
            k=args.k, k_min=args.min_k, k_max=args.max_k,
        )

        all_results.append(result)

        # Saving JSON results per config
        json_path = os.path.join(args.outdir, f"word2vec_bagofwords_{cfg['name']}.json")
        with open(json_path, "w") as f:
            json.dump(result, f, indent=2, default=str)
        print(f"  Saved JSON -> {json_path}")

        # Saving best results
        sil = result["metrics"]["cosine_silhouette"]
        if sil > best_silhouette:
            best_silhouette = sil
            best_result = result
        
    # Cross-configuration comparison
    plot_config_comparison(all_results, args.outdir)

    all_json_path = os.path.join(args.outdir, "word2vec_bagofwords_all_configs.json")
    with open(all_json_path, "w") as f:
        json.dump(
            {"recommended_configuration": best_result["config"]["name"],
             "results": all_results},
            f, indent=2, default=str,
        )
    print(f"Saved combined JSON -> {all_json_path}")

    if best_result is not None:
        cfg = best_result["config"]
        print("\n" + "=" * 80)
        print("  BEST CONFIGURATION (by cosine silhouette)")
        print("=" * 80)
        print(
            f"  {cfg['name']}  |  "
            f"bins={cfg['n_bins']}, "
            f"w2v_dim={cfg['w2v_vector_size']}  "
            f"(cosine silhouette = {best_silhouette:.4f})"
        )
        print("\nDone.")


if __name__ == "__main__":
    main()
