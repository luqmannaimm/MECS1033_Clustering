"""
K-Means Clustering (From Scratch) - 2D ONLY + Matplotlib Plot
-------------------------------------------------------------
Rules:
- You MUST implement the clustering algorithm yourself (no sklearn/scipy clustering).
- You MAY use pandas/numpy/matplotlib for loading data + plotting.

Recommended dataset (easy 2D):
- Mall Customers: use 2 columns:
  ["Annual Income (k$)", "Spending Score (1-100)"]

What this script does:
1) Load CSV
2) Pick exactly TWO numeric columns (2D clustering only)
3) Standardize the 2D features (important for distance-based clustering)
4) Run K-means from scratch (with k-means++ initialization implemented manually)
5) Plot the clustered points + centroids using matplotlib
6) Save output CSV with cluster labels

Beginner-friendly parameter reasons:
- Standardization: prevents one feature dominating distance.
- k rule-of-thumb: k ≈ sqrt(n/2) as a reasonable starting point; you can try multiple k.
- max_iters: safety stop.
"""

from __future__ import annotations

import math
import random
from typing import List, Tuple, Optional, Dict

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


# ----------------------------
# 1) Data loading + preparation (2D ONLY)
# ----------------------------

def load_csv_2d(
    csv_path: str,
    x_col: str,
    y_col: str
) -> Tuple[np.ndarray, pd.DataFrame]:
    """
    Load CSV and return X as a 2D feature matrix (n, 2).

    Args:
        csv_path: path to CSV file
        x_col: name of first feature column
        y_col: name of second feature column

    Returns:
        X: numpy array shape (n, 2)
        df: original dataframe
    """
    df = pd.read_csv(csv_path)

    if x_col not in df.columns or y_col not in df.columns:
        raise ValueError(
            f"Columns not found. Available columns:\n{list(df.columns)}\n"
            f"Requested: {x_col}, {y_col}"
        )

    # Ensure numeric
    X = df[[x_col, y_col]].to_numpy(dtype=float)
    if X.shape[1] != 2:
        raise ValueError("This script enforces exactly 2D features.")

    return X, df


def standardize_2d(X: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Standardize 2D features to mean=0, std=1 per dimension.

    Why:
    - K-means uses Euclidean distance. If one axis has larger scale, it dominates clustering.
    """
    mean = X.mean(axis=0)
    std = X.std(axis=0)
    std_safe = np.where(std == 0, 1.0, std)
    X_std = (X - mean) / std_safe
    return X_std, mean, std_safe


# ----------------------------
# 2) Distance + metrics helpers
# ----------------------------

def euclidean_distance_sq(a: np.ndarray, b: np.ndarray) -> float:
    """Squared Euclidean distance (faster; no sqrt)."""
    d = a - b
    return float(np.dot(d, d))


def compute_sse(X: np.ndarray, labels: np.ndarray, centroids: np.ndarray) -> float:
    """Sum of Squared Errors (within-cluster)."""
    sse = 0.0
    for i in range(X.shape[0]):
        c = labels[i]
        sse += euclidean_distance_sq(X[i], centroids[c])
    return sse


# ----------------------------
# 3) K-Means from scratch (2D)
# ----------------------------

def choose_k_rule_of_thumb(n: int) -> int:
    """
    Simple starting heuristic:
        k ≈ sqrt(n/2)

    You should still try multiple k and compare SSE / cluster separation for your report.
    """
    k = int(round(math.sqrt(n / 2.0)))
    return max(2, k)


def init_centroids_kmeanspp(X: np.ndarray, k: int, seed: int = 42) -> np.ndarray:
    """
    K-means++ initialization (implemented manually).

    Why:
    - More stable than pure random initialization.
    - Reduces chance of poor local minima.
    """
    rng = random.Random(seed)
    n, d = X.shape

    first_idx = rng.randrange(n)
    centroids = [X[first_idx].copy()]

    for _ in range(1, k):
        dist_sq = []
        for i in range(n):
            best = euclidean_distance_sq(X[i], centroids[0])
            for c in centroids[1:]:
                d2 = euclidean_distance_sq(X[i], c)
                if d2 < best:
                    best = d2
            dist_sq.append(best)

        total = sum(dist_sq)
        if total == 0:
            centroids.append(X[rng.randrange(n)].copy())
            continue

        r = rng.random() * total
        cum = 0.0
        chosen_idx = 0
        for i, w in enumerate(dist_sq):
            cum += w
            if cum >= r:
                chosen_idx = i
                break
        centroids.append(X[chosen_idx].copy())

    return np.vstack(centroids)


def assign_clusters(X: np.ndarray, centroids: np.ndarray) -> np.ndarray:
    """Assign each point to the nearest centroid."""
    n = X.shape[0]
    k = centroids.shape[0]
    labels = np.zeros(n, dtype=int)

    for i in range(n):
        best_c = 0
        best_d = euclidean_distance_sq(X[i], centroids[0])
        for c in range(1, k):
            d2 = euclidean_distance_sq(X[i], centroids[c])
            if d2 < best_d:
                best_d = d2
                best_c = c
        labels[i] = best_c
    return labels


def update_centroids(X: np.ndarray, labels: np.ndarray, k: int, seed: int = 42) -> np.ndarray:
    """
    Update centroids as mean of points in each cluster.
    Handles empty clusters by reseeding to a random point.
    """
    rng = random.Random(seed)
    _, d = X.shape
    centroids = np.zeros((k, d), dtype=float)

    for c in range(k):
        points = X[labels == c]
        if len(points) == 0:
            centroids[c] = X[rng.randrange(X.shape[0])]
        else:
            centroids[c] = points.mean(axis=0)

    return centroids


def kmeans_from_scratch(
    X: np.ndarray,
    k: int,
    max_iters: int = 100,
    tol: float = 1e-6,
    seed: int = 42
) -> Tuple[np.ndarray, np.ndarray, Dict[str, float]]:
    """
    K-means loop (from scratch).

    Stop conditions:
    - labels stop changing, OR
    - centroid movement is tiny (tol), OR
    - max_iters reached (safety)
    """
    n = X.shape[0]
    if k < 2 or k > n:
        raise ValueError(f"k must be in [2, n]. Got k={k}, n={n}")

    centroids = init_centroids_kmeanspp(X, k=k, seed=seed)
    labels = np.full(n, -1, dtype=int)

    for it in range(1, max_iters + 1):
        new_labels = assign_clusters(X, centroids)
        changes = int(np.sum(new_labels != labels))
        labels = new_labels

        new_centroids = update_centroids(X, labels, k=k, seed=seed)

        shift = float(np.sum((new_centroids - centroids) ** 2))
        centroids = new_centroids

        if changes == 0 or shift < tol:
            return labels, centroids, {
                "iterations": float(it),
                "label_changes_last_iter": float(changes),
                "centroid_shift_sq": float(shift),
            }

    return labels, centroids, {
        "iterations": float(max_iters),
        "label_changes_last_iter": float(changes),
        "centroid_shift_sq": float(shift),
    }


# ----------------------------
# 4) Plotting (matplotlib)
# ----------------------------

def plot_clusters_2d(
    X_raw: np.ndarray,
    labels: np.ndarray,
    centroids_raw: np.ndarray,
    x_label: str,
    y_label: str,
    title: str = "K-Means Clustering (2D)"
) -> None:
    """
    Plot 2D points colored by cluster, plus centroids.

    Note:
    - We plot using original (unstandardized) feature units for readability.
    """
    fig, ax = plt.subplots()
    scatter = ax.scatter(X_raw[:, 0], X_raw[:, 1], c=labels, s=25)
    ax.scatter(centroids_raw[:, 0], centroids_raw[:, 1], marker="X", s=200, c="black", edgecolors="w", linewidths=1.5, label="Centroids")
    # Draw a circle around each cluster
    for c in range(centroids_raw.shape[0]):
        cluster_points = X_raw[labels == c]
        if len(cluster_points) == 0:
            continue
        dists = np.linalg.norm(cluster_points - centroids_raw[c], axis=1)
        radius = dists.max()
        circle = plt.Circle((centroids_raw[c, 0], centroids_raw[c, 1]), radius, color=scatter.cmap(scatter.norm(c)), fill=False, linewidth=2, alpha=0.5, label=f"Cluster {c} boundary" if c == 0 else None)
        ax.add_patch(circle)
    ax.set_xlabel(x_label)
    ax.set_ylabel(y_label)
    ax.set_title(title)
    ax.legend(loc="best", fontsize="small")
    plt.tight_layout()
    out_path = "kmeans_2d_clustered_plot.png"
    plt.savefig(out_path, dpi=150)
    print(f"Saved cluster plot to: {out_path}")
    plt.close(fig)


# ----------------------------
# 5) Main (edit these settings)
# ----------------------------

def main():
    # ---- Dataset ----
    # Example: Mall Customers 2D clustering
    csv_path = "dataset/Mall_Customers.csv"
    x_col = "Annual Income (k$)"
    y_col = "Spending Score (1-100)"

    # ---- K-means parameters ----
    seed = 42
    max_iters = 100
    tol = 1e-6

    # Set k = 0 to auto-choose using rule-of-thumb; or set a fixed integer like 5.
    k = 0

    # ---- Load 2D data ----
    X_raw, df = load_csv_2d(csv_path, x_col=x_col, y_col=y_col)
    n = X_raw.shape[0]
    print(f"Loaded: {csv_path}")
    print(f"2D features: [{x_col}, {y_col}]  |  n={n}")

    # ---- Standardize for clustering ----
    X, mean, std = standardize_2d(X_raw)

    # ---- Choose k ----
    if k <= 0:
        k = choose_k_rule_of_thumb(n)
        print(f"Auto-chosen k={k} using k≈sqrt(n/2) heuristic.")
    else:
        print(f"Using user k={k}")

    # ---- Run K-means from scratch (on standardized space) ----
    labels, centroids_std, stats = kmeans_from_scratch(X, k=k, max_iters=max_iters, tol=tol, seed=seed)
    sse = compute_sse(X, labels, centroids_std)

    # Convert centroids back to original units for plotting/report readability
    centroids_raw = centroids_std * std + mean

    print("\n=== Results ===")
    print(f"Iterations: {int(stats['iterations'])}")
    print(f"Label changes (last iter): {int(stats['label_changes_last_iter'])}")
    print(f"Centroid shift (squared): {stats['centroid_shift_sq']:.6e}")
    print(f"SSE (in standardized space): {sse:.6f}")

    # ---- Save labeled output ----
    out_df = df.copy()
    out_df["cluster_id"] = labels
    out_path = "kmeans_2d_clustered_output.csv"
    out_df.to_csv(out_path, index=False)
    print(f"Saved labeled CSV: {out_path}")

    # ---- Plot ----
    plot_clusters_2d(
        X_raw=X_raw,
        labels=labels,
        centroids_raw=centroids_raw,
        x_label=x_col,
        y_label=y_col,
        title=f"K-Means (2D) | k={k}"
    )


if __name__ == "__main__":
    main()
