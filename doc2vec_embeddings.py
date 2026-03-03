"""Train Doc2Vec embeddings, cluster posts, and compare configurations."""

import os
import re
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
from gensim.models.doc2vec import Doc2Vec, TaggedDocument

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

MYSQL_HOST = os.environ.get("MYSQL_HOST", "localhost")
MYSQL_PORT = int(os.environ.get("MYSQL_PORT", "3306"))
MYSQL_USER = os.environ.get("MYSQL_USER", "root")
MYSQL_DATABASE_NAME = os.environ.get("MYSQL_DATABASE", "reddit_forum")
MYSQL_PASSWORD = os.environ.get("MYSQL_PASSWORD", "")


def get_conn():
    return pymysql.connect(
        host=MYSQL_HOST,
        port=MYSQL_PORT,
        user=MYSQL_USER,
        password=MYSQL_PASSWORD,
        database=MYSQL_DATABASE_NAME,
        charset="utf8mb4",
        cursorclass=pymysql.cursors.DictCursor,
    )


def load_posts(conn):
    """Load every post from the MySQL database."""
    query = """
    SELECT id, reddit_id, subreddit, cleaned_title, cleaned_selftext,
           keywords, title, selftext
    FROM posts
    ORDER BY id
    """
    with conn.cursor() as cur:
        cur.execute(query)
        rows = cur.fetchall()

    columns = [
        "id", "reddit_id", "subreddit", "cleaned_title",
        "cleaned_selftext", "keywords", "title", "selftext",
    ]
    df = pd.DataFrame(rows, columns=columns)
    df["id"] = pd.to_numeric(df["id"], errors="coerce")
    df["cleaned_title"] = df["cleaned_title"].fillna("")
    df["cleaned_selftext"] = df["cleaned_selftext"].fillna("")
    df["title"] = df["title"].fillna("")
    df["selftext"] = df["selftext"].fillna("")
    print(f"Loaded {len(df)} posts from MySQL.")
    return df


def build_documents(df):
    """Concatenate cleaned title and selftext into a single document string."""
    titles = df["cleaned_title"].fillna("")
    selftexts = df["cleaned_selftext"].fillna("")
    return (titles + ". " + selftexts).str.strip(". ").tolist()


def tokenize_docs(docs):
    """Whitespace tokenisation (text was already cleaned during ingestion)."""
    return [doc.lower().split() for doc in docs]


def train_doc2vec(tagged_docs, vector_size, min_count, epochs, dm=1, workers=4):
    """Build and train a Doc2Vec model."""
    print(
        f"  Training Doc2Vec: vector_size={vector_size}, "
        f"min_count={min_count}, epochs={epochs}, dm={dm}"
    )
    model = Doc2Vec(
        vector_size=vector_size,
        min_count=min_count,
        epochs=epochs,
        dm=dm,
        # Keep training stable and aligned with the reading's efficient setup.
        hs=0,
        negative=5,
        seed=42,
        workers=workers,
    )
    model.build_vocab(tagged_docs)
    model.train(tagged_docs, total_examples=len(tagged_docs), epochs=epochs)
    return model


def embeddings_from_model(model, num_docs):
    """Extract the document vectors from a trained Doc2Vec model."""
    return np.vstack([model.dv[i] for i in range(num_docs)])


def cosine_normalize(embeddings):
    """
    L2-normalise each embedding to unit length.
    After normalisation Euclidean distance is monotonically related to cosine
    distance, so KMeans on the normalised vectors effectively clusters by
    cosine similarity.
    """
    return normalize(embeddings, norm="l2")


def find_best_k(embeddings_norm, k_min=2, k_max=15):
    """
    Try every k in [k_min, k_max], run KMeans on L2-normalised embeddings,
    and pick the k that maximises the cosine silhouette score.
    """
    inertias, silhouettes = [], []
    k_values = list(range(k_min, k_max + 1))

    for k in k_values:
        km = KMeans(n_clusters=k, random_state=42, n_init=10)
        labels = km.fit_predict(embeddings_norm)
        inertias.append(km.inertia_)

        n_unique = len(np.unique(labels))
        if n_unique < 2:
            silhouettes.append(float("nan"))
            print(f"    k={k:>2d} | inertia={km.inertia_:>12.2f} | silhouette=N/A")
        else:
            sil = silhouette_score(embeddings_norm, labels, metric="cosine")
            silhouettes.append(sil)
            print(f"    k={k:>2d} | inertia={km.inertia_:>12.2f} | silhouette(cos)={sil:.4f}")

    valid = [(k, s) for k, s in zip(k_values, silhouettes) if not np.isnan(s)]
    best_k = max(valid, key=lambda x: x[1])[0] if valid else k_min
    print(f"  -> Best k = {best_k} (by cosine silhouette)")
    return best_k, k_values, inertias, silhouettes


def run_kmeans(embeddings_norm, k):
    """Fit KMeans on L2-normalised embeddings (cosine-equivalent clustering)."""
    km = KMeans(n_clusters=k, random_state=42, n_init=10)
    labels = km.fit_predict(embeddings_norm)
    return labels, km


def compute_metrics(embeddings_norm, labels, km):
    """Compute clustering metrics used in the report."""
    k = len(km.cluster_centers_)

    sil_overall = silhouette_score(embeddings_norm, labels, metric="cosine")

    sil_samples = silhouette_samples(embeddings_norm, labels, metric="cosine")
    per_cluster_sil = {}
    for cid in range(k):
        mask = labels == cid
        if mask.sum() > 0:
            per_cluster_sil[cid] = float(np.mean(sil_samples[mask]))
        else:
            per_cluster_sil[cid] = 0.0

    db = davies_bouldin_score(embeddings_norm, labels)
    ch = calinski_harabasz_score(embeddings_norm, labels)

    sizes = {int(cid): int(np.sum(labels == cid)) for cid in range(k)}
    size_std = float(np.std(list(sizes.values())))

    return {
        "cosine_silhouette": float(sil_overall),
        "per_cluster_silhouette": per_cluster_sil,
        "davies_bouldin": float(db),
        "calinski_harabasz": float(ch),
        "cluster_size_std": size_std,
        "cluster_sizes": sizes,
    }


def extract_cluster_keywords(df, labels, k, top_n=20):
    """
    For each cluster, combine all cleaned text, tokenise, remove NLTK
    stopwords, and return the top-n most frequent meaningful words.
    Returns {cluster_id: [(word, count), ...]}.
    """
    df = df.copy()
    df["cluster"] = labels
    kw_map = {}
    for cid in range(k):
        subset = df[df["cluster"] == cid]
        text = " ".join(
            (subset["cleaned_title"] + " " + subset["cleaned_selftext"]).tolist()
        ).lower()
        tokens = re.findall(r"[a-z]{3,}", text)
        words = [t for t in tokens if t not in STOPWORDS]
        kw_map[cid] = Counter(words).most_common(top_n)
    return kw_map


def nearest_to_centroid(embeddings_norm, labels, km, n=10):
    """
    For each cluster, find the n documents closest to the centroid using
    cosine distance and return their DataFrame indices.
    """
    closest = {}
    for cid, centroid in enumerate(km.cluster_centers_):
        mask = np.where(labels == cid)[0]
        if len(mask) == 0:
            closest[cid] = []
            continue
        dists = pairwise_distances(
            embeddings_norm[mask], centroid.reshape(1, -1), metric="cosine"
        ).ravel()
        order = np.argsort(dists)[:n]
        closest[cid] = mask[order].tolist()
    return closest


def subreddit_distribution(df, labels, k):
    """Return {cluster_id: {subreddit: count}} for every cluster."""
    df = df.copy()
    df["cluster"] = labels
    dist = {}
    for cid in range(k):
        dist[cid] = df[df["cluster"] == cid]["subreddit"].value_counts().to_dict()
    return dist


def plot_elbow_silhouettes(k_values, inertias, silhouettes, best_k, path):
    """Side-by-side elbow (inertia) and cosine-silhouette curves."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    ax1.plot(k_values, inertias, "o-", color="crimson")
    ax1.axvline(best_k, color="grey", ls="--", alpha=0.6, label=f"k={best_k}")
    ax1.set_xlabel("k (number of clusters)")
    ax1.set_ylabel("Inertia")
    ax1.set_title("Elbow Plot")
    ax1.grid(True, alpha=0.25)
    ax1.legend()

    sil_clean = [s if not np.isnan(s) else 0 for s in silhouettes]
    ax2.plot(k_values, sil_clean, "s-", color="forestgreen")
    ax2.axvline(best_k, color="grey", ls="--", alpha=0.6, label=f"k={best_k}")
    ax2.set_xlabel("k (number of clusters)")
    ax2.set_ylabel("Cosine Silhouette Score")
    ax2.set_title("Silhouette Scores (cosine)")
    ax2.grid(True, alpha=0.25)
    ax2.legend()

    plt.tight_layout()
    plt.savefig(path, dpi=150)
    plt.close()
    print(f"  Saved elbow/silhouette plot -> {path}")


def plot_clusters_2d(embeddings_norm, labels, k, kw_map, title_suffix, path):
    """PCA down to 2-D and colour-code by cluster assignment."""
    pca = PCA(n_components=2, random_state=42)
    coords = pca.fit_transform(embeddings_norm)
    cmap = plt.colormaps.get_cmap("tab10")

    fig, ax = plt.subplots(figsize=(14, 9))
    for cid in range(k):
        mask = labels == cid
        top_words = ", ".join(w for w, _ in kw_map.get(cid, [])[:5])
        ax.scatter(
            coords[mask, 0], coords[mask, 1],
            c=[cmap(cid % 10)],
            label=f"C{cid}: {top_words}",
            alpha=0.55, s=25, linewidths=0.25,
        )
    ax.set_xlabel(f"PC 1 ({pca.explained_variance_ratio_[0]*100:.1f}% var)")
    ax.set_ylabel(f"PC 2 ({pca.explained_variance_ratio_[1]*100:.1f}% var)")
    ax.set_title(f"2-D PCA Projection — {title_suffix}")
    ax.grid(True, alpha=0.2)
    ax.legend(fontsize=7, loc="best", ncol=1)
    plt.tight_layout()
    plt.savefig(path, dpi=150)
    plt.close()
    print(f"  Saved 2-D cluster plot -> {path}")


def plot_cluster_sizes(labels, k, kw_map, title_suffix, path):
    """Bar chart of the number of posts in each cluster."""
    counts = Counter(labels)
    ids = list(range(k))
    sizes = [counts.get(cid, 0) for cid in ids]
    bar_labels = [
        ", ".join(w for w, _ in kw_map.get(cid, [])[:3]) for cid in ids
    ]
    cmap = plt.colormaps.get_cmap("tab10")

    fig, ax = plt.subplots(figsize=(max(10, k * 1.4), 7))
    bars = ax.bar(ids, sizes, color=[cmap(i % 10) for i in ids], edgecolor="white")
    ax.set_xticks(ids)
    ax.set_xticklabels(
        [f"C{c}\n{lbl}" for c, lbl in zip(ids, bar_labels)],
        rotation=45, ha="center", fontsize=7,
    )
    ax.set_ylabel("Number of Posts")
    ax.set_title(f"Cluster Sizes — {title_suffix}")
    for bar, s in zip(bars, sizes):
        ax.text(
            bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.5,
            str(s), ha="center", va="bottom", fontsize=8,
        )
    plt.tight_layout()
    plt.savefig(path, dpi=150)
    plt.close()
    print(f"  Saved cluster-size plot -> {path}")


def plot_per_cluster_silhouettes(metrics, k, config_name, path):
    """Bar chart of per-cluster mean cosine silhouette scores."""
    pc_sil = metrics["per_cluster_silhouette"]
    ids = list(range(k))
    vals = [pc_sil.get(cid, 0.0) for cid in ids]
    cmap = plt.colormaps.get_cmap("tab10")

    fig, ax = plt.subplots(figsize=(max(8, k * 1.2), 5))
    bars = ax.bar(ids, vals, color=[cmap(i % 10) for i in ids], edgecolor="white")
    ax.axhline(metrics["cosine_silhouette"], color="black", ls="--", alpha=0.5,
               label=f"overall mean = {metrics['cosine_silhouette']:.4f}")
    ax.set_xticks(ids)
    ax.set_xticklabels([f"C{c}" for c in ids])
    ax.set_ylabel("Mean Cosine Silhouette")
    ax.set_title(f"Per-Cluster Silhouette — {config_name}")
    ax.grid(axis="y", alpha=0.25)
    ax.legend()
    for bar, v in zip(bars, vals):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.005,
                f"{v:.3f}", ha="center", va="bottom", fontsize=7)
    plt.tight_layout()
    plt.savefig(path, dpi=150)
    plt.close()
    print(f"  Saved per-cluster silhouette plot -> {path}")


def plot_config_comparison(results, outdir):
    """
    Generate a multi-panel comparison across all three configurations:
    silhouette, Davies-Bouldin, Calinski-Harabasz, and cluster-size balance.
    """
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

    plt.suptitle("Cross-Configuration Comparison", fontsize=14, fontweight="bold")
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    path = os.path.join(outdir, "config_comparison.png")
    plt.savefig(path, dpi=150)
    plt.close()
    print(f"\nSaved comparison plot -> {path}")


def run_configuration(df, docs, tokens, config, outdir,
                      k=None, k_min=2, k_max=15):
    """Run full pipeline for one Doc2Vec configuration."""
    name = config["name"]

    tagged_docs = [
        TaggedDocument(words=tokens[i], tags=[i]) for i in range(len(tokens))
    ]
    model = train_doc2vec(
        tagged_docs,
        vector_size=config["vector_size"],
        min_count=config["min_count"],
        epochs=config["epochs"],
        dm=config.get("dm", 1),
        workers=config.get("workers", 4),
    )

    raw_embeddings = embeddings_from_model(model, len(docs))
    embeddings_norm = cosine_normalize(raw_embeddings)
    print(f"  Embedding matrix shape: {embeddings_norm.shape} (L2-normalised)")

    if k is None:
        best_k, k_values, inertias, silhouettes = find_best_k(
            embeddings_norm, k_min=k_min, k_max=k_max
        )
    else:
        best_k = k
        k_values, inertias, silhouettes = None, None, None

    labels, km = run_kmeans(embeddings_norm, best_k)

    metrics = compute_metrics(embeddings_norm, labels, km)

    kw_map = extract_cluster_keywords(df, labels, best_k, top_n=20)
    nearest = nearest_to_centroid(embeddings_norm, labels, km, n=10)
    sub_dist = subreddit_distribution(df, labels, best_k)

    tag = f"{name} (dim={config['vector_size']}, mc={config['min_count']}, ep={config['epochs']}, dm={config['dm']})"
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

    print(f"\n{'-' * 70}")
    print(f"  DETAILED CLUSTER EXAMINATION — '{name}'")
    print(f"  Overall cosine silhouette = {metrics['cosine_silhouette']:.4f}")
    print(f"  Davies-Bouldin = {metrics['davies_bouldin']:.4f}  |  "
          f"Calinski-Harabasz = {metrics['calinski_harabasz']:.1f}")
    print(f"{'-' * 70}")

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

        for rank, idx in enumerate(nearest.get(cid, []), 1):
            row = df.iloc[idx]
            title = row["title"]
            selftext = row["selftext"]
            subreddit = row["subreddit"]
            print(f"      {rank}. [r/{subreddit}] {title}")
            if selftext.strip():
                print(f"         {selftext}")
            print("")
        print(f"  {'-' * 68}")

    cluster_summaries = []
    for cid in range(best_k):
        example_posts = []
        for idx in nearest.get(cid, []):
            row = df.iloc[idx]
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
            "per_cluster_silhouette": {str(k): v for k, v in metrics["per_cluster_silhouette"].items()},
            "davies_bouldin": metrics["davies_bouldin"],
            "calinski_harabasz": metrics["calinski_harabasz"],
            "cluster_size_std": metrics["cluster_size_std"],
            "cluster_sizes": metrics["cluster_sizes"],
        },
        "k_search": {
            "k_values": k_values,
            "inertias": [float(x) for x in inertias] if inertias else None,
            "silhouettes": (
                [float(x) if not np.isnan(x) else None for x in silhouettes]
                if silhouettes else None
            ),
        },
        "cluster_summaries": cluster_summaries,
    }


def main():
    parser = argparse.ArgumentParser(
        description="Lab 8 — Doc2Vec embeddings & clustering for Reddit posts"
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
        "--outdir", type=str, default="doc2vec_results",
        help="Directory for plots and JSON summaries.",
    )
    args = parser.parse_args()
    os.makedirs(args.outdir, exist_ok=True)

    conn = get_conn()
    try:
        df = load_posts(conn)
    finally:
        conn.close()

    if df.empty:
        print("No posts found in database. Run reddit_forum_analysis.py first.")
        return

    docs = build_documents(df)
    tokens = tokenize_docs(docs)
    non_empty_idx = [i for i, toks in enumerate(tokens) if toks]
    if not non_empty_idx:
        print("All posts are empty after preprocessing; nothing to embed.")
        return
    if len(non_empty_idx) < len(df):
        dropped = len(df) - len(non_empty_idx)
        print(f"Dropping {dropped} empty posts before Doc2Vec training.")
        df = df.iloc[non_empty_idx].reset_index(drop=True)
        docs = [docs[i] for i in non_empty_idx]
        tokens = [tokens[i] for i in non_empty_idx]

    print(f"Built {len(docs)} non-empty documents, vocabulary tokens ready.\n")

    configs = [
        {
            "name": "small",
            "vector_size": 50,
            "min_count": 2,
            "epochs": 30,
            "dm": 1,
        },
        {
            "name": "medium",
            "vector_size": 100,
            "min_count": 5,
            "epochs": 40,
            "dm": 1,
        },
        {
            "name": "large",
            "vector_size": 200,
            "min_count": 2,
            "epochs": 60,
            "dm": 0,
        },
    ]

    best_result = None
    best_silhouette = float("-inf")

    for cfg in configs:
        print("\n" + "=" * 80)
        print(f"  CONFIGURATION: {cfg['name'].upper()}")
        print(f"    vector_size = {cfg['vector_size']}")
        print(f"    min_count   = {cfg['min_count']}")
        print(f"    epochs      = {cfg['epochs']}")
        print(f"    dm          = {cfg['dm']}  "
              f"({'PV-DM (Distributed Memory)' if cfg['dm'] == 1 else 'PV-DBOW (Distributed Bag of Words)'})")
        print("=" * 80)

        result = run_configuration(
            df, docs, tokens, cfg, args.outdir,
            k=args.k, k_min=args.min_k, k_max=args.max_k,
        )

        sil = result["metrics"]["cosine_silhouette"]
        if sil > best_silhouette:
            best_silhouette = sil
            best_result = result

    if best_result is not None:
        cfg = best_result["config"]
        print("\n" + "=" * 80)
        print("  BEST CONFIGURATION (by cosine silhouette)")
        print("=" * 80)
        print(
            f"  {cfg['name']}  |  "
            f"dim={cfg['vector_size']}, min_count={cfg['min_count']}, "
            f"epochs={cfg['epochs']}, dm={cfg['dm']}  "
            f"(cosine silhouette = {best_silhouette:.4f})"
        )
        print("\nDone.")


if __name__ == "__main__":
    main()
