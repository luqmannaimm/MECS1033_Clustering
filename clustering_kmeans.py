# Subject       : MECS1033 Advanced Artificial Intelligence
# Task          : Assignment 2 - Clustering
# Script name   : clustering_kmeans.py
# Description   : K-Means Clustering on Seeds Dataset
# Author        : MEC255017 - luqmannaim@graduate.utm.my

from __future__ import annotations
from dataclasses import dataclass
from typing import List, Dict, Optional

import os
import math
import random
import itertools
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

#################
# CONFIGURATION #
#################

@dataclass
class KMeansConfig:
    """
    k           : choose automatically using rule-of-thumb
    max_iters   : max iterations. common safe default is 100
    tol         : centroid movement tolerance. 1e-6 is a common default
    seed        : random seed for centroid initialization. 42 chosen by default
    """
    k: Optional[int] = None
    max_iters: int = 100
    tol: float = 1e-6
    seed: int = 42

################
# LOAD DATASET #
################

def load_seeds(csv_path: str) -> pd.DataFrame:
    """Load Seeds dataset and extract 2D features"""

    # Read csv file into dataframe
    df = pd.read_csv(csv_path)
    
    # Check for missing columns
    required_cols = [
        "Area",
        "Perimeter",
        "Compactness",
        "Kernel Length",
        "Kernel Width",
        "Asymmetry Coefficient",
        "Kernel Groove Length"
    ]
    for c in required_cols:
        if c not in df.columns:
            raise ValueError(f"Missing column '{c}'. Found: {list(df.columns)}")


    # Return features and dataframe
    return df

########################
# STANDARDIZE FEATURES #
########################

def standardize_features(X: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Standardize features so both dimensions contribute equally to distance"""

    # Compute mean
    mean = X.mean(axis=0)

    # Compute safe standard deviation (avoid division by zero)
    std = X.std(axis=0)
    std_safe = np.where(std == 0, 1.0, std)

    # Standardize
    X_std = (X - mean) / std_safe

    # Return standardized features, mean, and safe standard deviation
    return X_std, mean, std_safe

######################
# DISTANCE FUNCTIONS #
######################

def compute_sed(a: np.ndarray, b: np.ndarray) -> float:
    """Compute Squared Euclidean Distance (SED) between two points"""

    # Compute difference
    diff = a - b

    # Compute SED
    sed = float(np.dot(diff, diff))

    # Return SED
    return sed

def compute_sse(X, labels, centroids) -> float:
    """Compute Sum of Squared Errors (SSE)"""

    # Initialize SSE
    sse = 0.0

    # Sum squared distances to assigned centroids
    for i in range(len(X)):
        sse += compute_sed(X[i], centroids[labels[i]])

    # Return SSE
    return sse

######################
# K-MEANS CLUSTERING #
######################

def choose_krot(n: int) -> int:
    """Choose K using rule of thumb (kROT)"""

    # Compute k using rule of thumb
    k = math.sqrt(n/2)

    # Round k to nearest integer
    k_round = int(round(k))

    # Ensure k is at least 2
    k_max = max(2, k_round)

    # Return chosen k
    return k_max

def init_centroids_kmeanspp(X: np.ndarray, k: int, seed: int = 42) -> np.ndarray:
    """
    Initialize centroids using K-Means++ algorithm
    - Normal K-Means: randomly selects k data points as initial centroids.
    - K-Means++: selects initial centroids to be distant from each other, improving convergence and clustering quality.
    """

    # Set random seed
    randseed = random.Random(seed)

    # Get number of data points
    n = len(X)

    # Initialize centroids list with one random point
    centroids = [X[randseed.randrange(n)].copy()]

    # Select remaining centroids
    for _ in range(1, k):

        # Compute squared distances from each point to the nearest centroid
        dist_sq = []
        for i in range(n):
            best = min(compute_sed(X[i], c) for c in centroids)
            dist_sq.append(best)

        # Choose next centroid with probability proportional to distance squared
        total = sum(dist_sq)
        r = randseed.random() * total
        
        # Select next furthermost centroid 
        cumulative_dist = 0.0
        for i, d in enumerate(dist_sq):
            cumulative_dist += d
            if cumulative_dist >= r:
                centroids.append(X[i].copy())
                break

    # Return centroids as array
    return np.vstack(centroids)

def assign_clusters(X: np.ndarray, centroids: np.ndarray) -> np.ndarray:
    """Assign each data point to the nearest centroid"""

    # Initialize labels array
    labels = np.zeros(len(X), dtype=int)

    # Assign each point to the nearest centroid
    for i in range(len(X)):
        labels[i] = min(
            range(len(centroids)),
            key=lambda c: compute_sed(X[i], centroids[c])
        )

    # Return updated labels
    return labels

def update_centroids(X: np.ndarray, labels: np.ndarray, k: int) -> np.ndarray:
    """Update centroids as the mean of assigned points"""

    # Initialize centroids array
    centroids = np.zeros((k, 2))

    # Go through each cluster
    for c in range(k):

        # Get points assigned to each cluster
        pts = X[labels == c]

        # Update centroid as mean of assigned points or random point if none assigned
        centroids[c] = pts.mean(axis=0) if len(pts) else X[random.randrange(len(X))]

    # Return updated centroids
    return centroids

def kmeans_clustering(
        X: np.ndarray,
        k: int,
        max_iters: int,
        tol: float,
        seed: int
    ) -> tuple[np.ndarray, np.ndarray]:
    """
    K-Means clustering algorithm
    X (np.ndarray)  : standardized data points
    k (int)         : number of clusters
    max_iters (int) : maximum number of iterations. 100 by default because it is common choice for k-means
    tol (float)     : tolerance for convergence. 1e-6 by default because it is common choice for k-means
    seed (int)      : random seed for centroid initialization. 42 by default for reproducibility
    """

    # Initialize centroids
    centroids = init_centroids_kmeanspp(X, k, seed=seed)

    # Initialize labels
    labels = np.full(len(X), -1, dtype=int)

    # Go through iterations
    for it in range(max_iters):

        # Assign clusters
        new_labels = assign_clusters(X, centroids)

        # Check for convergence of labels
        if np.array_equal(new_labels, labels):
            break
        labels = new_labels

        # Update centroids
        new_centroids = update_centroids(X, labels, k)

        # Check for convergence of centroids
        shift = np.sum((new_centroids - centroids) ** 2)
        centroids = new_centroids
        if shift < tol:
            break

    # Return final labels and centroids
    return labels, centroids

def run_kmeans(
        df: pd.DataFrame,
        xcol: str,
        ycol: str,
        cfg: KMeansConfig
    ) -> Dict:
    """
    Pipeline to run k-means:

    1) extract data
    2) standardize features
    3) choose k using rule-of-thumb (if not provided)
    4) run kmeans clustering
    5) revert centroid standardization for plotting
    6) compute SSE in standardized space
    """

    # Extract data
    X_raw = df[[xcol, ycol]].to_numpy(dtype=float)

    # Standardize features
    X_std, mean, std_safe = standardize_features(X_raw)

    # Choose k using rule of thumb (if not provided in cfg)
    k = cfg.k if cfg.k is not None else choose_krot(len(X_std))

    # Run k-means clustering
    labels, centroids_std = kmeans_clustering(
        X_std, k, max_iters=cfg.max_iters, tol=cfg.tol, seed=cfg.seed
    )

    # Revert centroid standardization for plotting
    centroids_raw = centroids_std * std_safe + mean

    # Compute SSE
    sse = compute_sse(X_std, labels, centroids_std)

    # Return results in a dict
    return {
        "xcol": xcol,
        "ycol": ycol,
        "k": k,
        "X_raw": X_raw,
        "X_std": X_std,
        "labels": labels,
        "centroids_std": centroids_std,
        "centroids_raw": centroids_raw,
        "mean": mean,
        "std": std_safe,
        "sse": sse
    }

#################
# PLOT CLUSTERS #
#################

def plot_clusters(result: Dict, out_path: str) -> None:
    """Define plot setup for clustering"""

    # Get data
    X = result["X_raw"]
    labels = result["labels"]
    centroids = result["centroids_raw"]
    xcol = result["xcol"]
    ycol = result["ycol"]
    k = result["k"]
    sse = result["sse"]

    # Create scatter plot
    fig, ax = plt.subplots()
    ax.scatter(X[:, 0], X[:, 1], c=labels, s=30)

    # Plot centroids
    ax.scatter(
        centroids[:, 0], centroids[:, 1],
        marker="X", s=200, c="black",
        edgecolors="white", linewidths=1.5,
        label="Centroids"
    )

    # Plot labels, titles, and settings
    ax.set_xlabel(xcol)
    ax.set_ylabel(ycol)
    ax.set_title(f"K-Means (k={k}) | {xcol} vs {ycol} | SSE={sse:.2f}")
    ax.legend(loc="best", fontsize="small")
    plt.tight_layout()

    # Save plot to file
    plt.savefig(out_path, dpi=150)
    plt.close(fig)

def run_pipeline(
        df: pd.DataFrame,
        cfg: KMeansConfig,
        out_prefix: str = "seeds_kmeans"
    ) -> None:
    """Run k-means clustering pipeline"""

    # Exclude non-numeric columns and cluster_id if present
    numeric_cols = [col for col in df.select_dtypes(include=[np.number]).columns if col != "cluster_id"]

    # Generate all possible pairs from dataset
    pairs = list(itertools.combinations(numeric_cols, 2))

    # Plot for each pair
    for xcol, ycol in pairs:

        # Setup the plot
        result = run_kmeans(df, xcol, ycol, cfg)

        # Define filename
        out_path = f"{out_prefix}_{xcol}_vs_{ycol}.png"

        # Generate and save plot
        plot_clusters(result, out_path)
        print(f"Saved cluster plot to: {out_path}")

#################
# MAIN FUNCTION #
#################

def main():

    # Initialize K-Means configuration
    cfg = KMeansConfig()
    print(f"\nUsing {cfg}")

    # Load csv dataset
    dataset_path = "dataset"
    csv_name = "seeds.csv"
    file_path = os.path.join(dataset_path, csv_name)
    df = load_seeds(file_path)
    print(f"\nLoaded {file_path}")

    # Run k-means pipeline
    print("\nRunning k-means clustering pipeline...")
    run_pipeline(df, cfg, out_prefix=(os.path.join(dataset_path, "seeds_kmeans")))
    print("Done!")

if __name__ == "__main__":
    main()
