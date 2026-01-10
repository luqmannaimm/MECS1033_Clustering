"""
K-Means Clustering (From Scratch) – Seeds Dataset (2D)
-----------------------------------------------------
Rules satisfied:
- NO sklearn / scipy / ML clustering libraries
- K-Means algorithm implemented manually
- pandas / numpy / matplotlib used ONLY for data handling & visualization

Dataset:
- Seeds Dataset (UCI)
- Objects: wheat kernels
- 2D features: Area vs Perimeter
"""

from __future__ import annotations

import math
import random
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


# ----------------------------
# 1) Load Seeds dataset (2D)
# ----------------------------

def load_seeds_2d(csv_path: str) -> tuple[np.ndarray, pd.DataFrame]:
    """
    Load Seeds dataset and extract 2D features:
    - Area
    - Perimeter
    """
    df = pd.read_csv(csv_path)

    required_cols = ["A", "P"]
    for c in required_cols:
        if c not in df.columns:
            raise ValueError(f"Missing column '{c}'. Found: {list(df.columns)}")

    X = df[required_cols].to_numpy(dtype=float)
    return X, df


def standardize_2d(X: np.ndarray):
    """
    Standardize features so both dimensions contribute equally to distance.
    """
    mean = X.mean(axis=0)
    std = X.std(axis=0)
    std_safe = np.where(std == 0, 1.0, std)
    return (X - mean) / std_safe, mean, std_safe


# ----------------------------
# 2) Distance + SSE
# ----------------------------

def euclidean_distance_sq(a: np.ndarray, b: np.ndarray) -> float:
    diff = a - b
    return float(np.dot(diff, diff))


def compute_sse(X, labels, centroids) -> float:
    sse = 0.0
    for i in range(len(X)):
        sse += euclidean_distance_sq(X[i], centroids[labels[i]])
    return sse


# ----------------------------
# 3) K-Means (FROM SCRATCH)
# ----------------------------

def choose_k_rule_of_thumb(n: int) -> int:
    """
    Rule of thumb:
        k ≈ sqrt(n / 2)
    """
    return max(2, int(round(math.sqrt(n / 2))))


def init_centroids_kmeanspp(X: np.ndarray, k: int, seed: int = 42) -> np.ndarray:
    rng = random.Random(seed)
    n = len(X)

    centroids = [X[rng.randrange(n)].copy()]

    for _ in range(1, k):
        dist_sq = []
        for i in range(n):
            best = min(euclidean_distance_sq(X[i], c) for c in centroids)
            dist_sq.append(best)

        total = sum(dist_sq)
        r = rng.random() * total
        cum = 0.0
        for i, d in enumerate(dist_sq):
            cum += d
            if cum >= r:
                centroids.append(X[i].copy())
                break

    return np.vstack(centroids)


def assign_clusters(X: np.ndarray, centroids: np.ndarray) -> np.ndarray:
    labels = np.zeros(len(X), dtype=int)
    for i in range(len(X)):
        labels[i] = min(
            range(len(centroids)),
            key=lambda c: euclidean_distance_sq(X[i], centroids[c])
        )
    return labels


def update_centroids(X: np.ndarray, labels: np.ndarray, k: int) -> np.ndarray:
    centroids = np.zeros((k, 2))
    for c in range(k):
        pts = X[labels == c]
        centroids[c] = pts.mean(axis=0) if len(pts) else X[random.randrange(len(X))]
    return centroids


def kmeans_from_scratch(X: np.ndarray, k: int, max_iters=100, tol=1e-6):
    centroids = init_centroids_kmeanspp(X, k)
    labels = np.full(len(X), -1)

    for it in range(max_iters):
        new_labels = assign_clusters(X, centroids)
        if np.array_equal(new_labels, labels):
            break
        labels = new_labels
        new_centroids = update_centroids(X, labels, k)

        shift = np.sum((new_centroids - centroids) ** 2)
        centroids = new_centroids
        if shift < tol:
            break

    return labels, centroids


# ----------------------------
# 4) Plotting (2D)
# ----------------------------


def plot_clusters_all_pairs(df, labels, centroids_raw, out_prefix="seeds_kmeans_2d"):
    """
    Plot clusters for all pairs of numeric features in the DataFrame.
    """
    import itertools
    # Exclude non-numeric columns and cluster_id if present
    numeric_cols = [col for col in df.select_dtypes(include=[np.number]).columns if col != "cluster_id"]
    pairs = list(itertools.combinations(numeric_cols, 2))
    for xcol, ycol in pairs:
        X = df[[xcol, ycol]].to_numpy(dtype=float)
        n = len(X)
        k = choose_k_rule_of_thumb(n)
        # Standardize for clustering
        mean = X.mean(axis=0)
        std = X.std(axis=0)
        std_safe = np.where(std == 0, 1.0, std)
        X_std = (X - mean) / std_safe
        # Run k-means for this pair
        labels_pair, centroids_std = kmeans_from_scratch(X_std, k)
        centroids = centroids_std * std + mean
        fig, ax = plt.subplots()
        scatter = ax.scatter(X[:, 0], X[:, 1], c=labels_pair, s=30)
        ax.scatter(centroids[:, 0], centroids[:, 1], marker="X", s=200, c="black", edgecolors="w", linewidths=1.5, label="Centroids")
        for c in range(centroids.shape[0]):
            cluster_points = X[labels_pair == c]
            if len(cluster_points) == 0:
                continue
            dists = np.linalg.norm(cluster_points - centroids[c], axis=1)
            radius = dists.max()
            circle = plt.Circle((centroids[c, 0], centroids[c, 1]), radius, color=scatter.cmap(scatter.norm(c)), fill=False, linewidth=2, alpha=0.5, label=f"Cluster {c} boundary" if c == 0 else None)
            ax.add_patch(circle)
        ax.set_xlabel(xcol)
        ax.set_ylabel(ycol)
        ax.set_title(f"K-Means: {xcol} vs {ycol}")
        ax.legend(loc="best", fontsize="small")
        plt.tight_layout()
        out_path = f"{out_prefix}_plot_{xcol}_vs_{ycol}.png"
        plt.savefig(out_path, dpi=150)
        print(f"Saved cluster plot to: {out_path}")
        plt.close(fig)


# ----------------------------
# 5) Main
# ----------------------------

def main():
    csv_path = "dataset/seeds.csv"   # <-- change if needed

    X_raw, df = load_seeds_2d(csv_path)
    X, mean, std = standardize_2d(X_raw)

    n = len(X)
    k = choose_k_rule_of_thumb(n)
    print(f"n={n}, chosen k={k}")

    labels, centroids_std = kmeans_from_scratch(X, k)
    centroids_raw = centroids_std * std + mean

    sse = compute_sse(X, labels, centroids_std)
    print(f"SSE (standardized space): {sse:.4f}")

    # Save output
    df_out = df.copy()
    df_out["cluster_id"] = labels
    df_out.to_csv("seeds_kmeans_2d_output.csv", index=False)

    plot_clusters_all_pairs(df, labels, centroids_raw)


if __name__ == "__main__":
    main()
