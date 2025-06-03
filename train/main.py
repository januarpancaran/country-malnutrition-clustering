import os
import pickle
import warnings

import numpy as np
import pandas as pd
from dbhc_model import DBHCClustering
from kmeans_model import KMeansClustering
from sklearn.decomposition import PCA
from sklearn.metrics import (calinski_harabasz_score, davies_bouldin_score,
                             silhouette_score)
from sklearn.preprocessing import StandardScaler

warnings.filterwarnings("ignore")


class NutritionClusteringAnalysis:
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
            ][
                :6
            ]  # Limit to 6 features max

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

    def run_analysis(self):
        """Main analysis pipeline with comprehensive error handling"""
        print("Starting Nutrition Clustering Analysis")
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
        analyzer = NutritionClusteringAnalysis()
        results = analyzer.run_analysis()

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
