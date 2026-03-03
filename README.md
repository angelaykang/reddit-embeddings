# Document Embeddings and Clustering

## Overview

This project builds document embeddings from Reddit posts and performs unsupervised clustering to analyze topic structure across posts.

The pipeline:
- loads preprocessed Reddit data from MySQL,
- trains three Doc2Vec configurations,
- clusters embeddings using a cosine-distance workflow,
- compares model configurations using quantitative and qualitative signals,
- exports plots and JSON summaries.

## Features

- Doc2Vec training with configurable hyperparameters
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

- `doc2vec_embeddings.py`: end-to-end embedding and clustering pipeline
- `requirements.txt`: Python dependencies for this project

## Requirements

- Python 3.10+
- MySQL server with Reddit data in `reddit_forum.posts`

## Installation

From the repository root:

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
pip install -r "labs/lab 8/requirements.txt"
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

Run with automatic `k` selection:

```bash
python "labs/lab 8/doc2vec_embeddings.py" --outdir "labs/lab 8/doc2vec_results"
```

Run with fixed `k`:

```bash
python "labs/lab 8/doc2vec_embeddings.py" --k 5 --outdir "labs/lab 8/doc2vec_results_k5"
```

### Command-Line Options

- `--k <int>`: fixed number of clusters (skip auto-search)
- `--min-k <int>`: minimum `k` for search (default: `2`)
- `--max-k <int>`: maximum `k` for search (default: `15`)
- `--outdir <path>`: output directory (default: `doc2vec_results`)

## Outputs

For each configuration:
- `doc2vec_<config>.json`
- `elbow_sil_<config>.png` (if auto-search is used)
- `clusters_2d_<config>.png`
- `cluster_sizes_<config>.png`
- `per_cluster_sil_<config>.png`

Combined:
- `config_comparison.png`
- `doc2vec_all_configs.json`

## Troubleshooting

- **MySQL auth error (`caching_sha2_password` / `sha256_password`)**
  ```bash
  pip install cryptography
  ```

- **No data loaded**
  Confirm database credentials and ensure the `posts` table contains rows.

- **Missing dependencies**
  Reinstall from `requirements.txt` inside an activated virtual environment.

## Reproducibility

Random seeds are set for key modeling components to improve run-to-run consistency. Minor variation can still occur across environments and package versions.

## License

For educational and research use.
