# MECS1033_Clustering
Subject: MECS1033 Advanced Artificial Intelligence
Author: MEC255017 - luqmannaim@graduate.utm.my
Assignment: Assignment 2 - Clustering

## Prerequisites
- Python 3.7 or higher

## Setup

### 1. Activate Virtual Environment
```bash
source _venv/Scripts/activate
```

### 2. Install Dependencies
```bash
pip install -r requirements.txt
```

## Run the Application
```bash
python clustering_kmeans.py
```

## What It Does

1. **Loads the dataset** from `dataset/seeds.csv`
2. **Standardizes features** to ensure equal contribution to distance calculations
3. **Automatically determines k** using the rule-of-thumb
4. **Performs K-Means clustering** with K-Means++ initialization for all feature pairs
5. **Generates visualizations** showing cluster assignments, centroids, SSE metric

## Output
The script generates PNG files in the `dataset/` folder with the naming pattern:
```
seeds_kmeans_{Feature1}_vs_{Feature2}.png
```

For example:
- `seeds_kmeans_Area_vs_Perimeter.png`
- `seeds_kmeans_Compactness_vs_Kernel Length.png`
- And all other feature combinations...

## Configuration

The clustering parameters in the `KMeansConfig` class can be modified:
- `k`: Number of clusters (default: auto-calculated)
- `max_iters`: Maximum iterations (default: 100)
- `tol`: Convergence tolerance (default: 1e-6)
- `seed`: Random seed for reproducibility (default: 42)

## Dataset Features

The Seeds dataset contains 7 features:
1. Area
2. Perimeter
3. Compactness
4. Kernel Length
5. Kernel Width
6. Asymmetry Coefficient
7. Kernel Groove Length
 