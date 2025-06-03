import numpy as np
from scipy.spatial.distance import cdist
from sklearn.cluster import DBSCAN
from sklearn.decomposition import PCA
from sklearn.metrics import (calinski_harabasz_score, davies_bouldin_score,
                             silhouette_score)
from sklearn.preprocessing import StandardScaler


class DBHCClustering:
    def __init__(self, data, scale_data=True, use_pca=True, pca_var=0.95):
        """
        Initialize with optional data scaling and PCA
        """
        self.data_raw = data
        self.n_samples = data.shape[0]
        self.scaler = StandardScaler() if scale_data else None
        self.pca = PCA(n_components=pca_var) if use_pca else None

        data_proc = data
        if scale_data:
            data_proc = self.scaler.fit_transform(data_proc)
        if use_pca:
            data_proc = self.pca.fit_transform(data_proc)
        self.data = data_proc
        self.n_samples = self.data.shape[0]

    def estimate_eps_min_samples_grid(self):
        """
        Grid search for best eps, min_samples, and metric using silhouette score
        """
        best_score = -1
        best_params = (0.5, 3, "euclidean")
        best_labels = None
        eps_candidates = np.linspace(0.2, 1.5, 20)  # finer grid, lower max
        min_samples_candidates = list(range(2, 11))  # up to 10
        metric_candidates = ["euclidean", "manhattan", "cosine"]

        for metric in metric_candidates:
            for eps in eps_candidates:
                for min_samples in min_samples_candidates:
                    dbscan = DBSCAN(eps=eps, min_samples=min_samples, metric=metric)
                    labels = dbscan.fit_predict(self.data)
                    n_clusters = len(set(labels)) - (1 if -1 in labels else 0)
                    if n_clusters < 2 or n_clusters >= self.n_samples:
                        continue
                    try:
                        score = silhouette_score(self.data, labels, metric=metric)
                        if score > best_score:
                            best_score = score
                            best_params = (eps, min_samples, metric)
                            best_labels = labels
                    except Exception:
                        continue
        return best_params, best_labels, best_score

    def dbhc_clustering(self, target_clusters=3):
        """
        Improved DBHC clustering with PCA and grid search for DBSCAN params
        """
        (best_eps, best_min_samples, best_metric), best_labels, best_score = (
            self.estimate_eps_min_samples_grid()
        )
        print(
            f"Best DBSCAN params: eps={best_eps:.3f}, min_samples={best_min_samples}, metric={best_metric}, silhouette={best_score:.4f}"
        )

        if best_labels is None:
            # fallback: single cluster
            labels = np.zeros(self.n_samples, dtype=int)
            clusters = [
                {
                    "points": self.data,
                    "indices": np.arange(self.n_samples),
                    "centroid": np.mean(self.data, axis=0),
                    "size": self.n_samples,
                }
            ]
            return labels, clusters

        # Build clusters from best_labels
        clusters = []
        for label in set(best_labels):
            if label == -1:
                continue
            indices = np.where(best_labels == label)[0]
            points = self.data[indices]
            clusters.append(
                {
                    "points": points,
                    "indices": indices,
                    "centroid": np.mean(points, axis=0),
                    "size": len(indices),
                    "silhouette": (
                        silhouette_score(
                            self.data,
                            np.where(best_labels == label, 1, 0),
                            metric=best_metric,
                        )
                        if len(indices) > 1
                        else -1
                    ),
                }
            )

        # Merge clusters if needed
        while len(clusters) > target_clusters:
            centroids = np.array([c["centroid"] for c in clusters])
            dist_matrix = cdist(centroids, centroids, metric=best_metric)
            np.fill_diagonal(dist_matrix, np.inf)
            i, j = np.unravel_index(np.argmin(dist_matrix), dist_matrix.shape)
            merged_indices = np.concatenate(
                [clusters[i]["indices"], clusters[j]["indices"]]
            )
            merged_points = self.data[merged_indices]
            merged_cluster = {
                "points": merged_points,
                "indices": merged_indices,
                "centroid": np.mean(merged_points, axis=0),
                "size": len(merged_indices),
                "silhouette": (
                    silhouette_score(
                        self.data,
                        np.isin(np.arange(self.n_samples), merged_indices).astype(int),
                        metric=best_metric,
                    )
                    if len(merged_indices) > 1
                    else -1
                ),
            }
            clusters = [c for idx, c in enumerate(clusters) if idx not in [i, j]]
            clusters.append(merged_cluster)

        # Assign labels
        labels = np.full(self.n_samples, -1, dtype=int)
        for idx, cluster in enumerate(clusters):
            labels[cluster["indices"]] = idx

        # Assign noise to nearest cluster
        if np.any(labels == -1):
            noise_idx = np.where(labels == -1)[0]
            noise_points = self.data[noise_idx]
            centroids = np.array([c["centroid"] for c in clusters])
            nearest = np.argmin(
                cdist(noise_points, centroids, metric=best_metric), axis=1
            )
            labels[noise_idx] = nearest

        # Print scores
        if len(np.unique(labels)) > 1:
            sil_score = silhouette_score(self.data, labels, metric=best_metric)
            db_score = davies_bouldin_score(self.data, labels)
            ch_score = calinski_harabasz_score(self.data, labels)
            print(f"Silhouette Score: {sil_score:.4f}")
            print(f"Davies-Bouldin Score: {db_score:.4f}")
            print(f"Calinski-Harabasz Score: {ch_score:.4f}")
        else:
            print("Silhouette Score: N/A (single cluster)")

        return labels, clusters
