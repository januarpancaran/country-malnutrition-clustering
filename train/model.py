import pickle

import numpy as np
import pandas as pd
from scipy.spatial.distance import cdist
from sklearn.cluster import DBSCAN, KMeans
from sklearn.preprocessing import StandardScaler


def load_and_preprocess_data():
    file_path = "dataset/country-wise-average.csv"
    df = pd.read_csv(file_path)
    features = [
        "Income Classification",
        "Severe Wasting",
        "Wasting",
        "Overweight",
        "Stunting",
        "Underweight",
        "U5 Population ('000s)",
    ]

    X = df[features].copy()
    X.fillna(X.mean(), inplace=True)
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    return df["Country"].values, X_scaled


# KMeans
def kmeans_clustering(X, n_clusters=3):
    kmeans = KMeans(n_clusters=n_clusters, random_state=42)
    kmeans.fit(X)
    return kmeans


# DBHC 1: Generate Eps Values
def generate_eps_values(X, m):
    distances = cdist(X, X, metric="euclidean")
    np.fill_diagonal(distances, np.inf)
    dist_2nn = np.sort(distances, axis=1)[:, 1]
    sorted_dist = np.sort(dist_2nn)
    j = int(np.sqrt(m))
    eps_values = []

    idx = j
    while idx < m:
        eps_values.append(sorted_dist[idx])
        idx += j
    return eps_values


# DBHC 2: Identify Primitive Clusters
def identify_primitive_clusters(X, eps_values):
    clusters = []
    remaining_points = X.copy()
    remaining_indices = np.arange(X.shape[0])

    for eps in sorted(eps_values):
        if len(remaining_points) == 0:
            break
        db = DBSCAN(eps=eps, min_samples=3, metric="euclidean").fit(remaining_points)
        labels = db.labels_
        unique_labels = set(labels) - {-1}
        for label in unique_labels:
            cluster_indices = remaining_indices[labels == label]
            if len(cluster_indices) > 0:
                clusters.append(
                    {
                        "points": X[cluster_indices],
                        "indices": cluster_indices,
                        "centroid": np.mean(X[cluster_indices], axis=0),
                    }
                )
        mask = labels == -1
        if np.sum(mask) == 0:
            break
        remaining_points = remaining_points[mask]
        remaining_indices = remaining_indices[mask]
    return clusters


# DBHC 3: Merge Clusters
def merge_clusters(clusters, k):
    while len(clusters) > k:
        centroids = np.array([c["centroid"] for c in clusters])
        dist_matrix = cdist(centroids, centroids, metric="euclidean")
        np.fill_diagonal(dist_matrix, np.inf)
        i, j = np.unravel_index(np.argmin(dist_matrix), dist_matrix.shape)
        new_cluster = {
            "points": np.vstack((clusters[i]["points"], clusters[j]["points"])),
            "indices": np.concatenate((clusters[i]["indices"], clusters[j]["indices"])),
            "centroid": np.mean(
                np.vstack((clusters[i]["points"], clusters[j]["points"])), axis=0
            ),
        }
        clusters = [c for idx, c in enumerate(clusters) if idx not in [i, j]]
        clusters.append(new_cluster)
    labels = np.full(X.shape[0], -1, dtype=int)
    for idx, cluster in enumerate(clusters):
        labels[cluster["indices"]] = idx
    return labels, clusters


# DBHC Core
def dbhc_clustering(X, k):
    m = X.shape[0]
    eps_values = generate_eps_values(X, m)
    clusters = identify_primitive_clusters(X, eps_values)
    labels, final_clusters = merge_clusters(clusters, k)
    return labels, final_clusters


countries, X = load_and_preprocess_data()

# KMeans
kmeans_model = kmeans_clustering(X)

# DBHC
dbhc_k = 3
dbhc_labels, dbhc_model = dbhc_clustering(X, dbhc_k)

# Save Results
kmeans_labels = kmeans_model.labels_
kmeans_result_df = pd.DataFrame(
    {
        "Country": countries,
        "Cluster": kmeans_labels,
    }
)
kmeans_result_df.to_csv("dataset/results/kmeans_clustering_results.csv", index=False)

dbhc_result_df = pd.DataFrame(
    {
        "Country": countries,
        "Cluster": dbhc_labels,
    }
)
dbhc_result_df.to_csv("dataset/results/dbhc_clustering_results.csv", index=False)

# Save Models with Pickle
with open("train/kmeans_model.pkl", "wb") as f:
    pickle.dump(kmeans_model, f)
with open("train/dbhc_model.pkl", "wb") as f:
    pickle.dump(dbhc_model, f)
