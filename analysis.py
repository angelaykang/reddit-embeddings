import os
import json
import glob
import argparse
from typing import Dict, List, Any

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


def safe_get(d: Dict, keys: List[str], default=None):
    cur = d
    for k in keys:
        if not isinstance(cur, dict) or k not in cur:
            return default
        cur = cur[k]
    return cur


def load_json(path: str) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def ensure_dir(path: str):
    os.makedirs(path, exist_ok=True)


def nice_method_name(method_key: str) -> str:
    if method_key == "doc2vec":
        return "Doc2Vec"
    if method_key == "word2vec_bow":
        return "Word2Vec BoW"
    return method_key


def method_sort_key(method_key: str) -> int:
    return 0 if method_key == "doc2vec" else 1


def print_simple_table(df: pd.DataFrame, float_cols=None):
    if df.empty:
        print("No rows to display.")
        return

    df_to_print = df.copy()

    if float_cols is None:
        float_cols = []

    for col in float_cols:
        if col in df_to_print.columns:
            df_to_print[col] = df_to_print[col].map(
                lambda x: f"{x:.4f}" if pd.notnull(x) else ""
            )

    headers = list(df_to_print.columns)
    rows = df_to_print.astype(str).values.tolist()

    col_widths = []
    for i, h in enumerate(headers):
        max_len = len(str(h))
        for row in rows:
            max_len = max(max_len, len(str(row[i])))
        col_widths.append(max_len)

    def format_row(row_vals):
        return " | ".join(str(v).ljust(col_widths[i]) for i, v in enumerate(row_vals))

    sep = "-+-".join("-" * w for w in col_widths)

    print(format_row(headers))
    print(sep)
    for row in rows:
        print(format_row(row))


def load_doc2vec_results(doc2vec_dir: str) -> List[Dict[str, Any]]:
    candidates = sorted(glob.glob(os.path.join(doc2vec_dir, "*.json")))
    results = []

    for path in candidates:
        data = load_json(path)
        if "config" in data and "metrics" in data:
            results.append({
                "method": "doc2vec",
                "path": path,
                "data": data,
            })

    return results


def load_word2vec_bow_results(word2vec_dir: str) -> List[Dict[str, Any]]:
    candidates = sorted(glob.glob(os.path.join(word2vec_dir, "*.json")))
    results = []

    for path in candidates:
        filename = os.path.basename(path)
        if "all_configs" in filename:
            continue
        data = load_json(path)
        if "config" in data and "metrics" in data:
            results.append({
                "method": "word2vec_bow",
                "path": path,
                "data": data,
            })

    return results


def extract_row(item: Dict[str, Any]) -> Dict[str, Any]:
    method = item["method"]
    data = item["data"]
    cfg = data["config"]
    metrics = data["metrics"]

    row = {
        "method_key": method,
        "method": nice_method_name(method),
        "config_name": cfg.get("name", "unknown"),
        "k": data.get("k"),
        "cosine_silhouette": metrics.get("cosine_silhouette"),
        "davies_bouldin": metrics.get("davies_bouldin"),
        "calinski_harabasz": metrics.get("calinski_harabasz"),
        "cluster_size_std": metrics.get("cluster_size_std"),
        "n_clusters": len(metrics.get("cluster_sizes", {})) if metrics.get("cluster_sizes") else None,
        "json_path": item["path"],
    }

    if method == "doc2vec":
        row["representation_dim"] = cfg.get("vector_size")
        row["min_count"] = cfg.get("min_count")
        row["epochs"] = cfg.get("epochs")
        row["variant"] = "PV-DM" if cfg.get("dm", 1) == 1 else "PV-DBOW"
        row["extra_info"] = (
            f"dim={cfg.get('vector_size')}, "
            f"min_count={cfg.get('min_count')}, "
            f"epochs={cfg.get('epochs')}, "
            f"dm={cfg.get('dm')}"
        )
    else:
        row["representation_dim"] = cfg.get("n_bins")
        row["min_count"] = cfg.get("w2v_min_count")
        row["epochs"] = None
        row["variant"] = f"W2V dim={cfg.get('w2v_vector_size')}"
        row["extra_info"] = (
            f"bins={cfg.get('n_bins')}, "
            f"w2v_dim={cfg.get('w2v_vector_size')}, "
            f"w2v_min_count={cfg.get('w2v_min_count')}"
        )

    return row


def build_summary_table(doc2vec_items: List[Dict], word2vec_items: List[Dict]) -> pd.DataFrame:
    rows = [extract_row(x) for x in (doc2vec_items + word2vec_items)]
    if not rows:
        return pd.DataFrame()

    df = pd.DataFrame(rows)
    df = df.sort_values(
        by=["method_key", "config_name"],
        key=lambda col: col.map(method_sort_key) if col.name == "method_key" else col
    ).reset_index(drop=True)
    return df


def choose_best_per_method(df: pd.DataFrame) -> pd.DataFrame:
    best_rows = []
    for method_key in sorted(df["method_key"].unique(), key=method_sort_key):
        sub = df[df["method_key"] == method_key].copy()
        sub = sub.sort_values(
            by=["cosine_silhouette", "davies_bouldin", "calinski_harabasz", "cluster_size_std"],
            ascending=[False, True, False, True]
        )
        best_rows.append(sub.iloc[0])
    return pd.DataFrame(best_rows).reset_index(drop=True)


def plot_metric_bars(summary_df: pd.DataFrame, outdir: str):
    df = summary_df.copy()
    df["label"] = df["method"].str.replace(" ", "\n") + "\n" + df["config_name"]

    metrics = [
        ("cosine_silhouette", "Cosine Silhouette (higher = better)", "all_configs_silhouette.png"),
        ("davies_bouldin", "Davies-Bouldin (lower = better)", "all_configs_davies_bouldin.png"),
        ("calinski_harabasz", "Calinski-Harabasz (higher = better)", "all_configs_calinski_harabasz.png"),
        ("cluster_size_std", "Cluster Size Std Dev (lower = better)", "all_configs_cluster_balance.png"),
    ]

    colors = []
    for _, row in df.iterrows():
        colors.append("#4C72B0" if row["method_key"] == "doc2vec" else "#DD8452")

    for metric, ylabel, filename in metrics:
        fig, ax = plt.subplots(figsize=(12, 6))
        vals = df[metric].values
        x = np.arange(len(df))

        bars = ax.bar(x, vals, color=colors, edgecolor="white")
        ax.set_xticks(x)
        ax.set_xticklabels(df["label"].tolist(), rotation=0, fontsize=9)
        ax.set_ylabel(ylabel)
        ax.set_title(f"Comparison Across All Configurations: {metric}")
        ax.grid(axis="y", alpha=0.25)

        offset = max(np.abs(vals).max() * 0.015, 0.003) if len(vals) > 0 else 0.01
        for bar, v in zip(bars, vals):
            if metric == "calinski_harabasz":
                txt = f"{v:.1f}"
            else:
                txt = f"{v:.4f}"
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                bar.get_height() + offset,
                txt,
                ha="center",
                va="bottom",
                fontsize=8,
            )

        plt.tight_layout()
        path = os.path.join(outdir, filename)
        plt.savefig(path, dpi=150)
        plt.close()
        print(f"Saved plot -> {path}")

def plot_method_average(summary_df: pd.DataFrame, outdir: str):
    grouped = summary_df.groupby(["method_key", "method"], as_index=False).agg({
        "cosine_silhouette": "mean",
        "davies_bouldin": "mean",
        "calinski_harabasz": "mean",
        "cluster_size_std": "mean",
    })

    metrics = [
        ("cosine_silhouette", "Mean Cosine Silhouette"),
        ("davies_bouldin", "Mean Davies-Bouldin"),
        ("calinski_harabasz", "Mean Calinski-Harabasz"),
        ("cluster_size_std", "Mean Cluster Size Std Dev"),
    ]

    fig, axes = plt.subplots(2, 2, figsize=(12, 9))
    axes = axes.ravel()
    colors = ["#4C72B0", "#DD8452"]

    for ax, (metric, title) in zip(axes, metrics):
        vals = grouped[metric].values
        names = grouped["method"].tolist()
        bars = ax.bar(names, vals, color=colors, edgecolor="white")
        ax.set_title(title)
        ax.grid(axis="y", alpha=0.25)

        offset = max(np.abs(vals).max() * 0.015, 0.003) if len(vals) > 0 else 0.01
        for bar, v in zip(bars, vals):
            if metric == "calinski_harabasz":
                txt = f"{v:.1f}"
            else:
                txt = f"{v:.4f}"
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                bar.get_height() + offset,
                txt,
                ha="center",
                va="bottom",
                fontsize=8,
            )

    plt.suptitle("Method-Level Average Comparison", fontsize=14, fontweight="bold")
    plt.tight_layout(rect=[0, 0, 1, 0.95])

    path = os.path.join(outdir, "method_level_average_comparison.png")
    plt.savefig(path, dpi=150)
    plt.close()
    print(f"Saved plot -> {path}")


def extract_best_item(items: List[Dict]) -> Dict[str, Any]:
    best_item = None
    best_sil = -1e18
    for x in items:
        sil = safe_get(x, ["data", "metrics", "cosine_silhouette"], float("-inf"))
        if sil > best_sil:
            best_sil = sil
            best_item = x
    return best_item


def summarize_cluster_themes(best_item: Dict[str, Any], top_k_clusters: int = 5, top_keywords: int = 8) -> str:
    if best_item is None:
        return "No best item found."

    data = best_item["data"]
    method_name = nice_method_name(best_item["method"])
    cfg = data["config"]
    k = data.get("k")
    cluster_summaries = data.get("cluster_summaries", [])

    cluster_summaries = sorted(cluster_summaries, key=lambda x: x.get("size", 0), reverse=True)

    lines = []
    lines.append(f"{method_name} qualitative cluster summary")
    lines.append(f"Best config: {cfg.get('name')} | k={k}")
    lines.append("-" * 60)

    for cluster in cluster_summaries[:top_k_clusters]:
        cid = cluster.get("cluster_id")
        size = cluster.get("size")
        sil = cluster.get("mean_cosine_silhouette")
        kws = cluster.get("top_keywords", [])[:top_keywords]
        kw_text = ", ".join([f"{x['word']}({x['count']})" for x in kws])

        sub_dist = cluster.get("subreddit_distribution", {})
        top_subs = sorted(sub_dist.items(), key=lambda x: -x[1])[:5]
        sub_text = ", ".join([f"{name}:{cnt}" for name, cnt in top_subs])

        rep_posts = cluster.get("representative_posts", [])[:3]
        lines.append(f"Cluster {cid} | size={size} | mean silhouette={sil:.4f}")
        lines.append(f"  Keywords: {kw_text}")
        lines.append(f"  Top subreddits: {sub_text}")
        lines.append("  Representative posts:")
        for idx, post in enumerate(rep_posts, 1):
            title = post.get("title", "")
            subreddit = post.get("subreddit", "")
            lines.append(f"    {idx}. [r/{subreddit}] {title}")
        lines.append("")

    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser(description="Section 3")
    parser.add_argument(
        "--doc2vec-dir", type=str, default="doc2vec_results",
        help="Directory containing Doc2Vec JSON result files."
    )
    parser.add_argument(
        "--word2vec-dir", type=str, default="word2vec_bagofwords_results",
        help="Directory containing Word2Vec BoW JSON result files."
    )
    parser.add_argument(
        "--outdir", type=str, default="section3_comparison_results",
        help="Directory to save plots and text outputs."
    )
    args = parser.parse_args()

    ensure_dir(args.outdir)

    doc2vec_items = load_doc2vec_results(args.doc2vec_dir)
    word2vec_items = load_word2vec_bow_results(args.word2vec_dir)

    if not doc2vec_items:
        print(f"WARNING: No Doc2Vec per-config JSON files found in {args.doc2vec_dir}")
    if not word2vec_items:
        print(f"WARNING: No Word2Vec BoW per-config JSON files found in {args.word2vec_dir}")

    summary_df = build_summary_table(doc2vec_items, word2vec_items)
    if summary_df.empty:
        print("No comparable results found. Exiting.")
        return

    best_df = choose_best_per_method(summary_df)

    plot_metric_bars(summary_df, args.outdir)
    plot_method_average(summary_df, args.outdir)

    best_doc_item = extract_best_item(doc2vec_items)
    best_bow_item = extract_best_item(word2vec_items)

    qualitative_text = []
    qualitative_text.append(summarize_cluster_themes(best_doc_item))
    qualitative_text.append("\n" + "=" * 72 + "\n")
    qualitative_text.append(summarize_cluster_themes(best_bow_item))

    qualitative_path = os.path.join(args.outdir, "best_config_qualitative_cluster_summary.txt")
    with open(qualitative_path, "w", encoding="utf-8") as f:
        f.write("\n".join(qualitative_text))
    print(f"Saved qualitative summary -> {qualitative_path}")

    print("\n" + "=" * 80)
    print("SECTION 3 COMPARABLE ANALYSIS SUMMARY")
    print("=" * 80)
    print_simple_table(
        summary_df[
            [
                "method",
                "config_name",
                "representation_dim",
                "k",
                "cosine_silhouette",
                "davies_bouldin",
                "calinski_harabasz",
                "cluster_size_std",
                "extra_info",
            ]
        ],
        float_cols=[
            "cosine_silhouette",
            "davies_bouldin",
            "calinski_harabasz",
            "cluster_size_std",
        ]
    )

    print("\n" + "-" * 80)
    print("BEST PER METHOD")
    print("-" * 80)
    print_simple_table(
        best_df[
            [
                "method",
                "config_name",
                "cosine_silhouette",
                "davies_bouldin",
                "calinski_harabasz",
                "cluster_size_std",
                "extra_info",
            ]
        ],
        float_cols=[
            "cosine_silhouette",
            "davies_bouldin",
            "calinski_harabasz",
            "cluster_size_std",
        ]
    )

    print("\nDone.")


if __name__ == "__main__":
    main()
