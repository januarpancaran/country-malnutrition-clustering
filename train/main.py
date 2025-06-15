import os
import pickle
import warnings

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from dbhc_model import DBHCClustering
from kmeans_model import KMeansClustering
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.metrics import (
    calinski_harabasz_score,
    davies_bouldin_score,
    silhouette_score,
)
from sklearn.preprocessing import StandardScaler

warnings.filterwarnings("ignore")


class MalnutritionClusteringAnalysis:
    def __init__(self, data_path="dataset/country-wise-average.csv"):
        self.data_path = data_path
        self.df = None
        self.countries = None
        self.X_processed = None
        self.scaler = None
        self.pca = None

    def check_data_quality(self, X, stage_name="Unknown"):
        print(f"\nData Quality Check - {stage_name}:")
        print(f"  Shape: {X.shape}")
        nan_count = (
            X.isnull().sum().sum() if hasattr(X, "isnull") else np.isnan(X).sum()
        )
        inf_count = (
            np.isinf(X.select_dtypes(include=[np.number]).values).sum()
            if hasattr(X, "select_dtypes")
            else np.isinf(X).sum()
        )
        print(f"  NaN values: {nan_count}")
        print(f"  Infinite values: {inf_count}")
        return nan_count == 0 and inf_count == 0

    def load_and_preprocess_data(self):
        """Load and preprocess the data with better error handling"""
        try:
            self.df = pd.read_csv(self.data_path)
            print(f"Loaded data with shape: {self.df.shape}")
            print(f"Columns: {list(self.df.columns)}")
        except FileNotFoundError:
            print(f"Data file not found at {self.data_path}")
            # Create sample data for testing
            print("Creating sample data for testing...")
            np.random.seed(42)
            n_countries = 50
            self.df = pd.DataFrame(
                {
                    "Country": [f"Country_{i}" for i in range(n_countries)],
                    "Income Classification": np.random.choice(
                        [
                            "Low income",
                            "Lower middle income",
                            "Upper middle income",
                            "High income",
                        ],
                        n_countries,
                    ),
                    "Severe Wasting": np.random.uniform(0, 15, n_countries),
                    "Wasting": np.random.uniform(5, 25, n_countries),
                    "Overweight": np.random.uniform(0, 20, n_countries),
                    "Stunting": np.random.uniform(10, 50, n_countries),
                    "Underweight": np.random.uniform(5, 35, n_countries),
                    "U5 Population ('000s)": np.random.uniform(100, 10000, n_countries),
                }
            )
            print("Sample data created successfully")

        # Income classification mapping
        income_map = {
            "Low income": 0,
            "Lower middle income": 1,
            "Upper middle income": 2,
            "High income": 3,
        }
        if "Income Classification" in self.df.columns:
            self.df["Income Classification"] = self.df["Income Classification"].map(
                income_map
            )
        else:
            self.df["Income Classification"] = 1

        # Fill numeric columns with mean
        numeric_cols = self.df.select_dtypes(include=[np.number]).columns
        self.df[numeric_cols] = self.df[numeric_cols].fillna(
            self.df[numeric_cols].mean()
        )

        # Feature engineering
        nutrition_cols = [
            "Severe Wasting",
            "Wasting",
            "Overweight",
            "Stunting",
            "Underweight",
        ]
        population_col = "U5 Population ('000s)"

        # Create per capita features if population data exists
        if population_col in self.df.columns:
            for col in nutrition_cols:
                if col in self.df.columns:
                    pop_values = self.df[population_col].fillna(1.0).replace(0, 1e-6)
                    self.df[f"{col} per capita"] = self.df[col].fillna(0.0) / pop_values

        # Create interaction features
        if "Stunting" in self.df.columns and "Underweight" in self.df.columns:
            self.df["Stunting_Underweight"] = self.df["Stunting"].fillna(0.0) * self.df[
                "Underweight"
            ].fillna(0.0)
        if "Wasting" in self.df.columns and "Overweight" in self.df.columns:
            self.df["Wasting_Overweight"] = self.df["Wasting"].fillna(0.0) * self.df[
                "Overweight"
            ].fillna(0.0)
        if "Severe Wasting" in self.df.columns:
            self.df["SevereWasting_Income"] = self.df["Severe Wasting"].fillna(
                0.0
            ) * self.df["Income Classification"].fillna(1.0)

        # Select features for clustering
        potential_features = [
            "Income Classification",
            "Severe Wasting per capita",
            "Wasting per capita",
            "Overweight per capita",
            "Stunting per capita",
            "Underweight per capita",
            "Stunting_Underweight",
            "Wasting_Overweight",
            "SevereWasting_Income",
        ]
        available_features = [
            col for col in potential_features if col in self.df.columns
        ]

        # Fallback to basic numeric features if engineered features don't exist
        if not available_features:
            available_features = [
                col
                for col in numeric_cols
                if col.lower() not in ["country", "year"]
                and not col.startswith("U5 Population")
            ][:6]  # Limit to 6 features max

        print(f"Using features: {available_features}")

        if not available_features:
            raise ValueError("No suitable features found for clustering")

        X = self.df[available_features].copy()
        X = X.fillna(X.mean()).fillna(0)
        self.check_data_quality(X, "After cleanup")

        # Extract country names
        self.countries = (
            self.df["Country"].values
            if "Country" in self.df.columns
            else np.array([f"Country_{i}" for i in range(len(self.df))])
        )

        # Scale and apply PCA
        self.scaler = StandardScaler()
        X_scaled = self.scaler.fit_transform(X)
        X_scaled = np.nan_to_num(X_scaled, nan=0.0, posinf=0.0, neginf=0.0)

        # Apply PCA with variance threshold
        n_components = min(0.95, X_scaled.shape[1] - 1, X_scaled.shape[0] - 1)
        self.pca = PCA(n_components=n_components)
        self.X_processed = self.pca.fit_transform(X_scaled)

        print(f"Final processed data shape: {self.X_processed.shape}")
        print(
            f"Explained variance ratio: {self.pca.explained_variance_ratio_.sum():.3f}"
        )

        return self.X_processed

    def plot_clustering_results_2clusters(
        self, labels, method_name="Clustering", figsize=(10, 8), save_plot=True
    ):
        """
        Create a 2D PCA projection plot for 2-cluster results

        Parameters:
        - labels: cluster labels (0s and 1s for 2 clusters)
        - method_name: name of clustering method for title
        - figsize: figure size tuple
        - save_plot: whether to save the plot to file
        """

        # Create PCA for visualization (separate from the main PCA used for clustering)
        pca_viz = PCA(n_components=2)
        X_pca = pca_viz.fit_transform(self.X_processed)
        explained_var = pca_viz.explained_variance_ratio_

        print(
            f"Visualization PCA - Explained variance: PC1={explained_var[0]:.3f}, PC2={explained_var[1]:.3f}"
        )

        # Define colors and labels for 2 clusters
        colors = ["#1f77b4", "#ff7f0e"]  # Blue and Orange
        cluster_labels = ["Cluster 0", "Cluster 1"]

        # Create the plot
        plt.figure(figsize=figsize)

        # Plot each cluster
        unique_labels = np.unique(labels)
        for i, label in enumerate(unique_labels):
            if label == -1:  # Handle noise points for DBSCAN-like algorithms
                mask = labels == label
                plt.scatter(
                    X_pca[mask, 0],
                    X_pca[mask, 1],
                    c="gray",
                    label="Noise/Outliers",
                    alpha=0.5,
                    s=40,
                    marker="x",
                )
            else:
                mask = labels == label
                if np.any(mask):
                    color_idx = min(i, len(colors) - 1)
                    plt.scatter(
                        X_pca[mask, 0],
                        X_pca[mask, 1],
                        c=colors[color_idx],
                        label=f"{cluster_labels[min(label, 1)]} (n={np.sum(mask)})",
                        alpha=0.7,
                        s=60,
                        edgecolors="white",
                        linewidth=0.5,
                    )

        # Calculate and plot centroids for valid clusters
        for label in unique_labels:
            if label != -1:
                mask = labels == label
                if np.any(mask):
                    centroid = np.mean(X_pca[mask], axis=0)
                    plt.scatter(
                        centroid[0],
                        centroid[1],
                        c="black",
                        s=200,
                        marker="x",
                        linewidth=3,
                    )

        # Customize the plot
        plt.xlabel(f"PCA Component 1 ({explained_var[0]:.1%} variance)", fontsize=12)
        plt.ylabel(f"PCA Component 2 ({explained_var[1]:.1%} variance)", fontsize=12)
        plt.title(
            f"{method_name} Results (PCA Projection)", fontsize=14, fontweight="bold"
        )
        plt.legend(bbox_to_anchor=(1.05, 1), loc="upper left")
        plt.grid(True, alpha=0.3)
        plt.tight_layout()

        # Save plot if requested
        if save_plot:
            try:
                os.makedirs("dataset/results/plots", exist_ok=True)
                filename = f"dataset/results/plots/{method_name.lower().replace(' ', '_')}_2clusters.png"
                plt.savefig(filename, dpi=300, bbox_inches="tight")
                print(f"Plot saved as: {filename}")
            except Exception as e:
                print(f"Could not save plot: {str(e)}")

        plt.show()

        return X_pca

    def plot_with_interpretation_2clusters(self, labels, method_name="Clustering"):
        """
        Enhanced version with cluster interpretation for 2 clusters
        """

        # Create PCA for visualization
        pca_viz = PCA(n_components=2)
        X_pca = pca_viz.fit_transform(self.X_processed)
        explained_var = pca_viz.explained_variance_ratio_

        # Define meaningful cluster names (customize based on your domain knowledge)
        cluster_names = ["Lower Malnutrition Risk", "Higher Malnutrition Risk"]
        colors = ["#2E8B57", "#DC143C"]  # Sea Green and Crimson

        # Create the plot
        plt.figure(figsize=(12, 8))

        # Plot each cluster
        unique_labels = np.unique(labels)
        centroids = []

        for i, label in enumerate(unique_labels):
            if label == -1:  # Handle noise points
                mask = labels == label
                plt.scatter(
                    X_pca[mask, 0],
                    X_pca[mask, 1],
                    c="gray",
                    label=f"Outliers (n={np.sum(mask)})",
                    alpha=0.5,
                    s=60,
                    marker="x",
                )
            else:
                mask = labels == label
                if np.any(mask):
                    color_idx = min(label, len(colors) - 1)
                    cluster_name = cluster_names[min(label, len(cluster_names) - 1)]

                    plt.scatter(
                        X_pca[mask, 0],
                        X_pca[mask, 1],
                        c=colors[color_idx],
                        label=f"{cluster_name} (n={np.sum(mask)})",
                        alpha=0.7,
                        s=80,
                        edgecolors="black",
                        linewidth=0.5,
                    )

                    # Calculate and mark centroid
                    centroid = np.mean(X_pca[mask], axis=0)
                    centroids.append(centroid)
                    plt.scatter(
                        centroid[0],
                        centroid[1],
                        c="black",
                        s=200,
                        marker="x",
                        linewidth=3,
                    )

        # Customize the plot
        plt.xlabel(f"PCA Component 1 ({explained_var[0]:.1%} variance)", fontsize=14)
        plt.ylabel(f"PCA Component 2 ({explained_var[1]:.1%} variance)", fontsize=14)
        plt.title(
            f"{method_name} Results - 2 Cluster Analysis",
            fontsize=16,
            fontweight="bold",
        )
        plt.legend(bbox_to_anchor=(1.05, 1), loc="upper left", fontsize=10)
        plt.grid(True, alpha=0.3)

        # Add cluster statistics
        stats_text = "Cluster Statistics:\n"
        for label in unique_labels:
            if label != -1:
                mask = labels == label
                count = np.sum(mask)
                percentage = (count / len(labels)) * 100
                cluster_name = cluster_names[min(label, len(cluster_names) - 1)]
                stats_text += f"{cluster_name}: {count} countries ({percentage:.1f}%)\n"

        plt.text(
            0.02,
            0.98,
            stats_text,
            transform=plt.gca().transAxes,
            verticalalignment="top",
            bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.8),
        )

        plt.tight_layout()

        # Save plot
        try:
            os.makedirs("dataset/results/plots", exist_ok=True)
            filename = f"dataset/results/plots/{method_name.lower().replace(' ', '_')}_interpretation.png"
            plt.savefig(filename, dpi=300, bbox_inches="tight")
            print(f"Interpretation plot saved as: {filename}")
        except Exception as e:
            print(f"Could not save interpretation plot: {str(e)}")

        plt.show()

        return X_pca, centroids

    def evaluate_clustering(self, labels, method_name):
        """Evaluate clustering quality with robust error handling"""
        unique_labels = len(set(labels))

        if unique_labels <= 1:
            print(f"{method_name}: Cannot evaluate - only one cluster found")
            return {"n_clusters": unique_labels}

        if unique_labels >= len(labels):
            print(f"{method_name}: Cannot evaluate - each point is its own cluster")
            return {"n_clusters": unique_labels}

        try:
            sil_score = silhouette_score(self.X_processed, labels)
            db_score = davies_bouldin_score(self.X_processed, labels)
            ch_score = calinski_harabasz_score(self.X_processed, labels)

            print(f"\n{method_name} Results:")
            print(f"  Silhouette Score: {sil_score:.4f}")
            print(f"  Davies-Bouldin Score: {db_score:.4f}")
            print(f"  Calinski-Harabasz Score: {ch_score:.4f}")
            print(f"  Number of clusters: {unique_labels}")

            return {
                "silhouette": sil_score,
                "davies_bouldin": db_score,
                "calinski_harabasz": ch_score,
                "n_clusters": unique_labels,
            }
        except Exception as e:
            print(f"Error evaluating {method_name}: {str(e)}")
            return {"n_clusters": unique_labels, "error": str(e)}

    def plot_optimal_k_analysis(self, max_k=10, save_plots=True):
        """
        Visualize different methods for finding optimal K

        Parameters:
        - max_k: maximum number of clusters to test
        - save_plots: whether to save plots to file
        """
        if self.X_processed is None:
            print("No processed data available. Run load_and_preprocess_data() first.")
            return None

        print(f"\nAnalyzing optimal K from 1 to {max_k}...")

        # Initialize storage for metrics
        k_range = range(1, max_k + 1)
        inertias = []
        silhouette_scores = []
        davies_bouldin_scores = []
        calinski_harabasz_scores = []

        # Calculate metrics for each K
        for k in k_range:
            print(f"Testing K={k}...")

            if k == 1:
                # Handle single cluster case
                inertias.append(
                    np.sum((self.X_processed - np.mean(self.X_processed, axis=0)) ** 2)
                )
                silhouette_scores.append(0)  # Undefined for single cluster
                davies_bouldin_scores.append(0)
                calinski_harabasz_scores.append(0)
            else:
                # Run K-means
                kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
                labels = kmeans.fit_predict(self.X_processed)

                # Calculate metrics
                inertias.append(kmeans.inertia_)

                try:
                    sil_score = silhouette_score(self.X_processed, labels)
                    db_score = davies_bouldin_score(self.X_processed, labels)
                    ch_score = calinski_harabasz_score(self.X_processed, labels)

                    silhouette_scores.append(sil_score)
                    davies_bouldin_scores.append(db_score)
                    calinski_harabasz_scores.append(ch_score)
                except Exception as e:
                    print(f"Error calculating metrics for K={k}: {e}")
                    silhouette_scores.append(0)
                    davies_bouldin_scores.append(0)
                    calinski_harabasz_scores.append(0)

        # Create comprehensive visualization
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        fig.suptitle(
            "Optimal K Analysis for Clustering", fontsize=16, fontweight="bold"
        )

        # 1. Elbow Method (Inertia)
        axes[0, 0].plot(k_range, inertias, "bo-", linewidth=2, markersize=8)
        axes[0, 0].set_xlabel("Number of Clusters (K)", fontsize=12)
        axes[0, 0].set_ylabel("Inertia (Within-cluster sum of squares)", fontsize=12)
        axes[0, 0].set_title("Elbow Method", fontsize=14, fontweight="bold")
        axes[0, 0].grid(True, alpha=0.3)

        # Find and mark elbow point
        if len(inertias) > 2:
            # Simple elbow detection using the "knee" point
            diffs = np.diff(inertias)
            second_diffs = np.diff(diffs)
            if len(second_diffs) > 0:
                elbow_idx = np.argmax(second_diffs) + 2  # +2 because of double diff
                if elbow_idx < len(k_range):
                    axes[0, 0].axvline(
                        x=k_range[elbow_idx],
                        color="red",
                        linestyle="--",
                        label=f"Elbow at K={k_range[elbow_idx]}",
                    )
                    axes[0, 0].legend()

        # 2. Silhouette Score
        axes[0, 1].plot(
            k_range[1:], silhouette_scores[1:], "go-", linewidth=2, markersize=8
        )
        axes[0, 1].set_xlabel("Number of Clusters (K)", fontsize=12)
        axes[0, 1].set_ylabel("Silhouette Score", fontsize=12)
        axes[0, 1].set_title("Silhouette Analysis", fontsize=14, fontweight="bold")
        axes[0, 1].grid(True, alpha=0.3)

        # Mark best silhouette score
        if len(silhouette_scores[1:]) > 0:
            best_sil_idx = np.argmax(silhouette_scores[1:]) + 1
            best_sil_k = k_range[best_sil_idx]
            axes[0, 1].axvline(
                x=best_sil_k,
                color="red",
                linestyle="--",
                label=f"Best K={best_sil_k} (Score: {silhouette_scores[best_sil_idx]:.3f})",
            )
            axes[0, 1].legend()

        # 3. Davies-Bouldin Score (lower is better)
        axes[1, 0].plot(
            k_range[1:], davies_bouldin_scores[1:], "ro-", linewidth=2, markersize=8
        )
        axes[1, 0].set_xlabel("Number of Clusters (K)", fontsize=12)
        axes[1, 0].set_ylabel("Davies-Bouldin Score", fontsize=12)
        axes[1, 0].set_title(
            "Davies-Bouldin Analysis (Lower is Better)", fontsize=14, fontweight="bold"
        )
        axes[1, 0].grid(True, alpha=0.3)

        # Mark best Davies-Bouldin score
        if len(davies_bouldin_scores[1:]) > 0:
            best_db_idx = (
                np.argmin([score for score in davies_bouldin_scores[1:] if score > 0])
                + 1
            )
            if best_db_idx < len(davies_bouldin_scores):
                best_db_k = k_range[best_db_idx]
                axes[1, 0].axvline(
                    x=best_db_k,
                    color="red",
                    linestyle="--",
                    label=f"Best K={best_db_k} (Score: {davies_bouldin_scores[best_db_idx]:.3f})",
                )
                axes[1, 0].legend()

        # 4. Calinski-Harabasz Score (higher is better)
        axes[1, 1].plot(
            k_range[1:], calinski_harabasz_scores[1:], "mo-", linewidth=2, markersize=8
        )
        axes[1, 1].set_xlabel("Number of Clusters (K)", fontsize=12)
        axes[1, 1].set_ylabel("Calinski-Harabasz Score", fontsize=12)
        axes[1, 1].set_title(
            "Calinski-Harabasz Analysis (Higher is Better)",
            fontsize=14,
            fontweight="bold",
        )
        axes[1, 1].grid(True, alpha=0.3)

        # Mark best Calinski-Harabasz score
        if len(calinski_harabasz_scores[1:]) > 0:
            best_ch_idx = np.argmax(calinski_harabasz_scores[1:]) + 1
            best_ch_k = k_range[best_ch_idx]
            axes[1, 1].axvline(
                x=best_ch_k,
                color="red",
                linestyle="--",
                label=f"Best K={best_ch_k} (Score: {calinski_harabasz_scores[best_ch_idx]:.1f})",
            )
            axes[1, 1].legend()

        plt.tight_layout()

        # Save plot if requested
        if save_plots:
            try:
                os.makedirs("dataset/results/plots", exist_ok=True)
                filename = "dataset/results/plots/optimal_k_analysis.png"
                plt.savefig(filename, dpi=300, bbox_inches="tight")
                print(f"Optimal K analysis plot saved as: {filename}")
            except Exception as e:
                print(f"Could not save optimal K plot: {str(e)}")

        plt.show()

        # Print summary of recommendations
        print("\n" + "=" * 50)
        print("OPTIMAL K RECOMMENDATIONS")
        print("=" * 50)

        if len(silhouette_scores[1:]) > 0:
            best_sil_k = k_range[np.argmax(silhouette_scores[1:]) + 1]
            print(
                f"Best Silhouette Score: K={best_sil_k} (Score: {max(silhouette_scores[1:]):.4f})"
            )

        if len(davies_bouldin_scores[1:]) > 0:
            valid_db_scores = [
                score for score in davies_bouldin_scores[1:] if score > 0
            ]
            if valid_db_scores:
                best_db_k = k_range[davies_bouldin_scores.index(min(valid_db_scores))]
                print(
                    f"Best Davies-Bouldin Score: K={best_db_k} (Score: {min(valid_db_scores):.4f})"
                )

        if len(calinski_harabasz_scores[1:]) > 0:
            best_ch_k = k_range[np.argmax(calinski_harabasz_scores[1:]) + 1]
            print(
                f"Best Calinski-Harabasz Score: K={best_ch_k} (Score: {max(calinski_harabasz_scores[1:]):.2f})"
            )

        # Return the metrics for further analysis
        return {
            "k_range": list(k_range),
            "inertias": inertias,
            "silhouette_scores": silhouette_scores,
            "davies_bouldin_scores": davies_bouldin_scores,
            "calinski_harabasz_scores": calinski_harabasz_scores,
            "recommended_k": {
                "silhouette": best_sil_k if len(silhouette_scores[1:]) > 0 else None,
                "davies_bouldin": best_db_k
                if len(davies_bouldin_scores[1:]) > 0 and valid_db_scores
                else None,
                "calinski_harabasz": best_ch_k
                if len(calinski_harabasz_scores[1:]) > 0
                else None,
            },
        }

    def plot_silhouette_analysis_detailed(self, k_range=None, save_plots=True):
        """
        Detailed silhouette analysis with individual cluster silhouette plots

        Parameters:
        - k_range: range of K values to analyze (default: [2, 3, 4, 5])
        - save_plots: whether to save plots to file
        """
        if self.X_processed is None:
            print("No processed data available. Run load_and_preprocess_data() first.")
            return None

        if k_range is None:
            k_range = [2, 3, 4, 5]

        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        fig.suptitle("Detailed Silhouette Analysis", fontsize=16, fontweight="bold")
        axes = axes.ravel()

        silhouette_avgs = []

        for idx, k in enumerate(k_range[:4]):  # Limit to 4 subplots
            # Perform clustering
            kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
            cluster_labels = kmeans.fit_predict(self.X_processed)

            # Calculate silhouette score
            silhouette_avg = silhouette_score(self.X_processed, cluster_labels)
            silhouette_avgs.append(silhouette_avg)

            # Calculate silhouette scores for each sample
            from sklearn.metrics import silhouette_samples

            sample_silhouette_values = silhouette_samples(
                self.X_processed, cluster_labels
            )

            ax = axes[idx]
            y_lower = 10

            for i in range(k):
                # Aggregate silhouette scores for samples belonging to cluster i
                ith_cluster_silhouette_values = sample_silhouette_values[
                    cluster_labels == i
                ]
                ith_cluster_silhouette_values.sort()

                size_cluster_i = ith_cluster_silhouette_values.shape[0]
                y_upper = y_lower + size_cluster_i

                color = plt.cm.nipy_spectral(float(i) / k)
                ax.fill_betweenx(
                    np.arange(y_lower, y_upper),
                    0,
                    ith_cluster_silhouette_values,
                    facecolor=color,
                    edgecolor=color,
                    alpha=0.7,
                )

                # Label the silhouette plots with their cluster numbers at the middle
                ax.text(-0.05, y_lower + 0.5 * size_cluster_i, str(i))
                y_lower = y_upper + 10

            ax.set_xlabel("Silhouette Coefficient Values")
            ax.set_ylabel("Cluster Label")
            ax.set_title(f"K={k}, Avg Score={silhouette_avg:.3f}")

            # Add vertical line for average silhouette score
            ax.axvline(
                x=silhouette_avg,
                color="red",
                linestyle="--",
                label=f"Avg: {silhouette_avg:.3f}",
            )
            ax.legend()

            ax.set_ylim([0, len(self.X_processed) + (k + 1) * 10])

        plt.tight_layout()

        # Save plot if requested
        if save_plots:
            try:
                os.makedirs("dataset/results/plots", exist_ok=True)
                filename = "dataset/results/plots/detailed_silhouette_analysis.png"
                plt.savefig(filename, dpi=300, bbox_inches="tight")
                print(f"Detailed silhouette analysis plot saved as: {filename}")
            except Exception as e:
                print(f"Could not save detailed silhouette plot: {str(e)}")

        plt.show()

        return silhouette_avgs

    def save_results(self, kmeans_labels, dbhc_labels, kmeans_model, dbhc_clusters):
        """Save results with proper directory creation"""
        try:
            os.makedirs("dataset/results", exist_ok=True)
            os.makedirs("train/models", exist_ok=True)

            # Save clustering results
            pd.DataFrame({"Country": self.countries, "Cluster": kmeans_labels}).to_csv(
                "dataset/results/kmeans_clustering_results.csv", index=False
            )
            pd.DataFrame({"Country": self.countries, "Cluster": dbhc_labels}).to_csv(
                "dataset/results/dbhc_clustering_results.csv", index=False
            )

            # Save models
            with open("train/models/kmeans_model.pkl", "wb") as f:
                pickle.dump(
                    {"model": kmeans_model, "scaler": self.scaler, "pca": self.pca}, f
                )
            with open("train/models/dbhc_model.pkl", "wb") as f:
                pickle.dump(
                    {"clusters": dbhc_clusters, "scaler": self.scaler, "pca": self.pca},
                    f,
                )

            print("\nResults saved successfully!")
            print("Files saved:")
            print("  - dataset/results/kmeans_clustering_results.csv")
            print("  - dataset/results/dbhc_clustering_results.csv")
            print("  - train/models/kmeans_model.pkl")
            print("  - train/models/dbhc_model.pkl")

        except Exception as e:
            print(f"Error saving results: {str(e)}")

    def run_optimal_k_analysis(self, max_k=10):
        """Run comprehensive optimal K analysis with all visualizations"""
        print("Starting Optimal K Analysis")
        print("=" * 50)

        try:
            # Load and preprocess data
            self.load_and_preprocess_data()

            if self.X_processed is None or len(self.X_processed) == 0:
                raise ValueError("No processed data available for clustering")

            # Run optimal K analysis
            print("\n" + "=" * 30)
            print("OPTIMAL K ANALYSIS")
            print("=" * 30)

            k_metrics = self.plot_optimal_k_analysis(max_k=max_k)

            print("\n" + "=" * 30)
            print("DETAILED SILHOUETTE ANALYSIS")
            print("=" * 30)

            silhouette_avgs = self.plot_silhouette_analysis_detailed()

            return {"k_metrics": k_metrics, "silhouette_averages": silhouette_avgs}

        except Exception as e:
            print(f"Error during optimal K analysis: {str(e)}")
            import traceback

            traceback.print_exc()
            return None

    def run_2cluster_analysis(self):
        """Run analysis specifically for 2 clusters with visualizations"""
        print("Starting 2-Cluster Malnutrition Analysis")
        print("=" * 50)

        try:
            # Load and preprocess data
            self.load_and_preprocess_data()

            if self.X_processed is None or len(self.X_processed) == 0:
                raise ValueError("No processed data available for clustering")

            # Run K-Means clustering with 2 clusters
            print("\n" + "=" * 30)
            print("KMEANS 2-CLUSTER ANALYSIS")
            print("=" * 30)

            kmeans_2 = KMeans(n_clusters=2, random_state=42, n_init=10)
            kmeans_2_labels = kmeans_2.fit_predict(self.X_processed)
            kmeans_2_metrics = self.evaluate_clustering(kmeans_2_labels, "KMeans-2")

            # Plot K-means results
            print("\nGenerating K-Means visualization...")
            self.plot_clustering_results_2clusters(kmeans_2_labels, "K-Means 2-Cluster")
            self.plot_with_interpretation_2clusters(
                kmeans_2_labels, "K-Means Malnutrition"
            )

            # Run DBHC clustering with 2 clusters
            print("\n" + "=" * 30)
            print("DBHC 2-CLUSTER ANALYSIS")
            print("=" * 30)

            dbhc_clustering = DBHCClustering(self.X_processed)
            dbhc_labels, dbhc_clusters = dbhc_clustering.dbhc_clustering(
                target_clusters=2
            )
            dbhc_2_metrics = self.evaluate_clustering(dbhc_labels, "DBHC-2")

            # Plot DBHC results
            print("\nGenerating DBHC visualization...")
            self.plot_clustering_results_2clusters(dbhc_labels, "DBHC 2-Cluster")
            self.plot_with_interpretation_2clusters(dbhc_labels, "DBHC Malnutrition")

            # Save results
            self.save_results(kmeans_2_labels, dbhc_labels, kmeans_2, dbhc_clusters)

            # Summary
            print("\n" + "=" * 50)
            print("2-CLUSTER ANALYSIS SUMMARY")
            print("=" * 50)
            print(f"Data points processed: {len(self.X_processed)}")
            print(f"Features after PCA: {self.X_processed.shape[1]}")
            print(f"K-Means clusters: {kmeans_2_metrics.get('n_clusters', 'N/A')}")
            print(f"DBHC clusters: {dbhc_2_metrics.get('n_clusters', 'N/A')}")

            if "silhouette" in kmeans_2_metrics and "silhouette" in dbhc_2_metrics:
                better_method = (
                    "K-Means"
                    if kmeans_2_metrics["silhouette"] > dbhc_2_metrics["silhouette"]
                    else "DBHC"
                )
                print(f"Better silhouette score: {better_method}")
                print(f"K-Means silhouette: {kmeans_2_metrics['silhouette']:.4f}")
                print(f"DBHC silhouette: {dbhc_2_metrics['silhouette']:.4f}")

            return {
                "kmeans_model": kmeans_2,
                "kmeans_labels": kmeans_2_labels,
                "kmeans_metrics": kmeans_2_metrics,
                "dbhc_labels": dbhc_labels,
                "dbhc_clusters": dbhc_clusters,
                "dbhc_metrics": dbhc_2_metrics,
                "optimal_k": 2,
            }

        except Exception as e:
            print(f"Error during 2-cluster analysis: {str(e)}")
            import traceback

            traceback.print_exc()
            return None

    def run_analysis(self):
        """Main analysis pipeline with comprehensive error handling"""
        print("Starting Malnutrition Clustering Analysis")
        print("=" * 50)

        try:
            # Load and preprocess data
            self.load_and_preprocess_data()

            if self.X_processed is None or len(self.X_processed) == 0:
                raise ValueError("No processed data available for clustering")

            # Run K-Means clustering
            print("\n" + "=" * 30)
            print("KMEANS CLUSTERING")
            print("=" * 30)

            kmeans_clustering = KMeansClustering(self.X_processed)
            kmeans_model, optimal_k = kmeans_clustering.find_optimal_kmeans()
            kmeans_labels = kmeans_model.labels_
            kmeans_metrics = self.evaluate_clustering(kmeans_labels, "KMeans")

            # Run DBHC clustering
            print("\n" + "=" * 30)
            print("DBHC CLUSTERING")
            print("=" * 30)

            dbhc_clustering = DBHCClustering(self.X_processed)
            dbhc_labels, dbhc_clusters = dbhc_clustering.dbhc_clustering(
                target_clusters=optimal_k
            )
            dbhc_metrics = self.evaluate_clustering(dbhc_labels, "DBHC")

            # Save results
            self.save_results(kmeans_labels, dbhc_labels, kmeans_model, dbhc_clusters)

            # Summary
            print("\n" + "=" * 50)
            print("ANALYSIS SUMMARY")
            print("=" * 50)
            print(f"Data points processed: {len(self.X_processed)}")
            print(f"Features after PCA: {self.X_processed.shape[1]}")
            print(f"K-Means clusters: {kmeans_metrics.get('n_clusters', 'N/A')}")
            print(f"DBHC clusters: {dbhc_metrics.get('n_clusters', 'N/A')}")

            if "silhouette" in kmeans_metrics and "silhouette" in dbhc_metrics:
                better_method = (
                    "K-Means"
                    if kmeans_metrics["silhouette"] > dbhc_metrics["silhouette"]
                    else "DBHC"
                )
                print(f"Better silhouette score: {better_method}")

            return {
                "kmeans_model": kmeans_model,
                "kmeans_labels": kmeans_labels,
                "kmeans_metrics": kmeans_metrics,
                "dbhc_labels": dbhc_labels,
                "dbhc_clusters": dbhc_clusters,
                "dbhc_metrics": dbhc_metrics,
                "optimal_k": optimal_k,
            }

        except Exception as e:
            print(f"Error during analysis: {str(e)}")
            import traceback

            traceback.print_exc()
            return None


def main():
    """Main function with comprehensive error handling"""
    try:
        analyzer = MalnutritionClusteringAnalysis()

        # Ask user which analysis to run
        print("Choose analysis type:")
        print("1. Standard analysis (finds optimal K)")
        print("2. 2-cluster analysis with visualizations")
        print("3. Optimal K analysis with visualizations")
        print("4. Complete analysis (optimal K + 2-cluster)")

        choice = input("Enter choice (1-4, default=2): ").strip()

        if choice == "1":
            results = analyzer.run_analysis()
        elif choice == "3":
            results = analyzer.run_optimal_k_analysis()
        elif choice == "4":
            print("\n" + "=" * 60)
            print("RUNNING COMPLETE ANALYSIS")
            print("=" * 60)

            # First run optimal K analysis
            k_results = analyzer.run_optimal_k_analysis()

            # Then run 2-cluster analysis
            cluster_results = analyzer.run_2cluster_analysis()

            results = {
                "optimal_k_analysis": k_results,
                "cluster_analysis": cluster_results,
            }
        else:  # Default to 2-cluster analysis
            results = analyzer.run_2cluster_analysis()

        if results:
            print("\nAnalysis completed successfully!")
        else:
            print("\nAnalysis failed - check error messages above")

        return results

    except Exception as e:
        print(f"Critical error in main: {str(e)}")
        import traceback

        traceback.print_exc()
        return None


if __name__ == "__main__":
    results = main()
