from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score


class KMeansClustering:
    def __init__(self, data):
        self.data = data

    def find_optimal_kmeans(self, k_range=(2, 10)):
        """Find optimal number of clusters for KMeans using silhouette score."""
        best_score = -1
        best_k = 2
        best_model = None
        scores = {}

        for k in range(k_range[0], k_range[1] + 1):
            model = KMeans(n_clusters=k, random_state=42, n_init=10)
            labels = model.fit_predict(self.data)

            if len(set(labels)) > 1:  # Ensure we have multiple clusters
                score = silhouette_score(self.data, labels)
                scores[k] = score

                if score > best_score:
                    best_score = score
                    best_k = k
                    best_model = model

        print(f"Optimal K for KMeans: {best_k} (Silhouette Score: {best_score:.4f})")
        print("Silhouette scores for different K values:")
        for k, score in scores.items():
            print(f"  K={k}: {score:.4f}")

        return best_model, best_k
