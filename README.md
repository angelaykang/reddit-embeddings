# Document Embeddings and Clustering

## Overview

This project builds document embeddings from Reddit posts and performs unsupervised clustering to analyze topic structure across posts. The two embedding methods implemented and compared are:

1. Doc2Vec (`doc2vec_embeddings.py`) learns document vectors directly via distributed memory / distributed bag-of-words.
2. Word2Vec Bag-of-Words (`word2vec_bagofwords_embeddings.py`) trains Word2Vec on all words, clusters word vectors into semantic bins, then represents each document as a noramlized bin-frequency vector.

Both pipelines:
- load preprocessed Reddit data from MySQL,
- train three embedding configurations at different dimensionalities (50, 100, 200),
- cluster embeddings using a cosine-distance workflow (L2 Normalization + KMeans),
- compare configurations using quantitative and qualitative signals,
- export plots for visual inspection of clustering quality.

## Features

- Doc2Vec training with configurable hyperparameters
- Word2Vec Bag-of-Words embedding with configurable bin counts and Word2Vec parameters
- Three embedding configurations evaluated in a single run
- Cosine-oriented clustering via L2 normalization + KMeans
- Automatic cluster-count selection with silhouette-based search
- Clustering metrics:
  - cosine silhouette (overall and per cluster)
  - Davies-Bouldin index
  - Calinski-Harabasz index
  - cluster size standard deviation
- Cluster interpretability outputs:
  - top keywords per cluster
  - subreddit distribution per cluster
  - representative posts nearest cluster centroids
- Visualization outputs:
  - elbow and silhouette curves
  - 2D PCA cluster map
  - cluster size chart
  - cross-configuration comparison plot

## Project Files

- `doc2vec_embeddings.py`: end-to-end embedding and clustering pipeline (Section 1)
- `word2vec_bagofwords_embeddings.py`: Word2Vec Bag-of-Words embedding and clustering pipeline (Section 2)
- `requirements.txt`: Python dependencies for this project

## Requirements

- Python 3.10+
- MySQL server with Reddit data in `reddit_forum.posts`

## Installation

From the repository root (this `lab 8` directory):

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
pip install -r requirements.txt
```

## Configuration

Set database environment variables before running:

```bash
export MYSQL_HOST=localhost
export MYSQL_PORT=3306
export MYSQL_USER=root
export MYSQL_PASSWORD="YOUR_PASSWORD"
export MYSQL_DATABASE=reddit_forum
```

## Usage

### Section 1: Doc2Vec Embeddings

Run with automatic `k` selection:

```bash
python doc2vec_embeddings.py --outdir "doc2vec_results"
```

Run with fixed `k`:

```bash
python doc2vec_embeddings.py --k 5 --outdir "doc2vec_results_k5"
```

#### Command-Line Options

- `--k <int>`: fixed number of clusters (skip auto-search)
- `--min-k <int>`: minimum `k` for search (default: `2`)
- `--max-k <int>`: maximum `k` for search (default: `15`)
- `--outdir <path>`: output directory (default: `doc2vec_results`)

### Section 2: Word2Vec Bag-of-Words Embeddings

Run with automatic `k` selection:

```bash
python word2vec_bagofwords_embeddings.py --outdir "word2vec_bagofwords_results"
```

Run with fixed `k`:

```bash
python word2vec_bagofwords_embeddings.py --k 5 --outdir "word2vec_bagofwords_results"
```

### Command-Line Options
`--k <int>`: fixed number of clusters (skip auto-search)
`--min-k <int>`: minimum `k` for search (default: `2`)
`--max-k <int>`: maximum `k` for search (default: `15`)
`--outdir <path>`: output directory (default: `word2vec_bagofwords_results`)
`--w2v-dim <int>`: Word2Vec vector dimensionality used for word binning (default: `150`)
`--w2v-min-count <int>`: Word2Vec minimum word frequency (default: `2`)
`--w2v-epochs <int>`: Word2Vec training epochs (default: `30`)

## Outputs

For each configuration:
- `elbow_sil_<config>.png` (if auto-search is used)
- `clusters_2d_<config>.png`
- `cluster_sizes_<config>.png`
- `per_cluster_sil_<config>.png`
- `config_comparison.png` (cross-configuration comparison)
- Per-config JSON summaries with metrics, cluster keywords, subreddit distributions, and representative posts
- Combined JSON with all configs and the recommended configuration


## Section 3: Comparative Analysis
