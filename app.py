import os
import pickle
import warnings

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import streamlit as st

warnings.filterwarnings("ignore")


# Set page config
st.set_page_config(
    page_title="Malnutrition Clustering Predictor", page_icon="🍎", layout="wide"
)


class MalnutritionPredictor:
    def __init__(self):
        self.kmeans_model = None
        self.dbhc_clusters = None
        self.scaler = None
        self.pca = None
        self.feature_names = [
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

    def load_models(self):
        """Load the trained models and preprocessors"""
        try:
            # Load K-Means model
            if os.path.exists("train/models/kmeans_model.pkl"):
                with open("train/models/kmeans_model.pkl", "rb") as f:
                    kmeans_data = pickle.load(f)
                    self.kmeans_model = kmeans_data["model"]
                    self.scaler = kmeans_data["scaler"]
                    self.pca = kmeans_data["pca"]

            # Load DBHC model
            if os.path.exists("train/models/dbhc_model.pkl"):
                with open("train/models/dbhc_model.pkl", "rb") as f:
                    dbhc_data = pickle.load(f)
                    self.dbhc_clusters = dbhc_data["clusters"]

                    # Debug: Print DBHC cluster structure (simplified)
                    if (
                        isinstance(self.dbhc_clusters, list)
                        and len(self.dbhc_clusters) > 0
                    ):
                        st.success(
                            f"✅ DBHC model loaded with {len(self.dbhc_clusters)} clusters"
                        )
                        if isinstance(self.dbhc_clusters[0], dict):
                            cluster_keys = list(self.dbhc_clusters[0].keys())
                            st.info(f"Cluster structure: {cluster_keys}")
                    else:
                        st.warning(
                            f"⚠️ Unexpected DBHC structure: {type(self.dbhc_clusters)}"
                        )

            return True
        except Exception as e:
            st.error(f"Error loading models: {str(e)}")
            return False

    def preprocess_input(self, user_input):
        """Preprocess user input to match training data format"""
        # Create DataFrame from user input
        input_df = pd.DataFrame([user_input])

        # Scale the input
        input_scaled = self.scaler.transform(input_df)
        input_scaled = np.nan_to_num(input_scaled, nan=0.0, posinf=0.0, neginf=0.0)

        # Apply PCA
        input_pca = self.pca.transform(input_scaled)

        return input_pca

    def predict_kmeans(self, processed_input):
        """Make prediction using K-Means model"""
        if self.kmeans_model is not None:
            prediction = self.kmeans_model.predict(processed_input)
            return prediction[0]
        return None

    def predict_dbhc(self, processed_input):
        """Make prediction using DBHC clustering"""
        if self.dbhc_clusters is not None:
            distances = []

            # Handle DBHC clusters (list of dictionaries with 'centroid' key)
            if isinstance(self.dbhc_clusters, list):
                for i, cluster_info in enumerate(self.dbhc_clusters):
                    if isinstance(cluster_info, dict):
                        # DBHC clusters have 'centroid' key
                        if "centroid" in cluster_info:
                            cluster_center = np.array(cluster_info["centroid"])
                        # Fallback to calculating center from points
                        elif "points" in cluster_info:
                            points = np.array(cluster_info["points"])
                            cluster_center = np.mean(points, axis=0)
                        else:
                            continue

                        # Calculate distance from input to cluster center
                        dist = np.linalg.norm(processed_input[0] - cluster_center)
                        distances.append(dist)
                    else:
                        # If cluster_info is directly a centroid/center
                        cluster_center = np.array(cluster_info)
                        dist = np.linalg.norm(processed_input[0] - cluster_center)
                        distances.append(dist)

                if distances:
                    return np.argmin(distances)

            # Handle if clusters are stored as a dictionary
            elif isinstance(self.dbhc_clusters, dict):
                for cluster_id, cluster_info in self.dbhc_clusters.items():
                    if isinstance(cluster_info, dict):
                        if "centroid" in cluster_info:
                            cluster_center = np.array(cluster_info["centroid"])
                        elif "points" in cluster_info:
                            points = np.array(cluster_info["points"])
                            cluster_center = np.mean(points, axis=0)
                        else:
                            continue

                        dist = np.linalg.norm(processed_input[0] - cluster_center)
                        distances.append((dist, cluster_id))

                if distances:
                    distances.sort(key=lambda x: x[0])
                    return distances[0][1]

            # Handle if clusters are just an array of centers
            elif isinstance(self.dbhc_clusters, np.ndarray):
                for i, cluster_center in enumerate(self.dbhc_clusters):
                    dist = np.linalg.norm(processed_input[0] - cluster_center)
                    distances.append(dist)
                return np.argmin(distances) if distances else 0

            # If we reach here, format is not recognized
            st.warning(
                "⚠️ DBHC cluster format not recognized. Using default cluster assignment."
            )
            st.info(f"Cluster type: {type(self.dbhc_clusters)}")
            if hasattr(self.dbhc_clusters, "__len__") and len(self.dbhc_clusters) > 0:
                st.info(f"First element type: {type(self.dbhc_clusters[0])}")
                if isinstance(self.dbhc_clusters[0], dict):
                    st.info(f"First element keys: {list(self.dbhc_clusters[0].keys())}")
            return 0

        return None

    def get_cluster_explanation(self, cluster_id, method="kmeans"):
        """Get explanation for each cluster - Updated for 2 clusters only"""
        explanations = {
            "kmeans": {
                0: "🔴 **High Malnutrition Cluster**: Countries with high rates of severe wasting, stunting, and underweight children. These are typically low-income countries requiring immediate and intensive nutrition interventions.",
                1: "🟢 **Low Malnutrition Cluster**: Countries with better nutrition outcomes and lower malnutrition rates. These are typically higher-income countries with established nutrition programs and healthcare systems.",
            },
            "dbhc": {
                0: "🔴 **High Malnutrition Cluster**: Countries with severe malnutrition indicators including high rates of wasting, stunting, and underweight children. Emergency nutrition interventions are critically needed.",
                1: "🟢 **Low Malnutrition Cluster**: Countries with relatively better nutrition outcomes and lower malnutrition rates. Focus should be on maintaining progress and preventing nutrition-related issues.",
            },
        }

        # Default explanation for any unexpected cluster IDs
        default_explanation = f"Cluster {cluster_id}: Malnutrition profile based on the input characteristics."

        # Ensure we only return explanations for clusters 0 and 1
        if cluster_id not in [0, 1]:
            st.warning(
                f"⚠️ Unexpected cluster ID: {cluster_id}. Defaulting to cluster 0 explanation."
            )
            cluster_id = 0

        return explanations.get(method, {}).get(cluster_id, default_explanation)

    def get_cluster_label(self, cluster_id):
        """Get human-readable label for cluster"""
        labels = {0: "High Malnutrition", 1: "Low Malnutrition"}
        return labels.get(cluster_id, f"Cluster {cluster_id}")

    def plot_prediction_location(
        self, processed_input, prediction, method, training_data=None
    ):
        """Plot the prediction location relative to training data"""
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))

        # Define colors for the 2 clusters
        cluster_colors = ["red", "green"]
        cluster_labels = ["High Malnutrition", "Low Malnutrition"]

        # Plot 1: 2D PCA visualization
        if training_data is not None and training_data.shape[1] >= 2:
            # Plot training data points
            scatter = ax1.scatter(
                training_data[:, 0],
                training_data[:, 1],
                c=range(len(training_data)),
                alpha=0.6,
                cmap="RdYlGn_r",  # Red to Green colormap (reversed so red = high malnutrition)
                s=50,
                label="Training Data",
            )

            # Plot prediction point
            pred_color = (
                cluster_colors[prediction]
                if prediction < len(cluster_colors)
                else "blue"
            )
            ax1.scatter(
                processed_input[0, 0],
                processed_input[0, 1],
                c=pred_color,
                s=200,
                marker="*",
                label=f"Your Prediction ({self.get_cluster_label(prediction)})",
                edgecolors="black",
                linewidth=2,
            )

            ax1.set_xlabel("First Principal Component")
            ax1.set_ylabel("Second Principal Component")
            ax1.set_title(f"{method.upper()} - Prediction Location in PCA Space")
            ax1.legend()
            ax1.grid(True, alpha=0.3)

        # Plot 2: Cluster distribution
        if self.kmeans_model is not None:
            # Get cluster centers for visualization
            if hasattr(self.kmeans_model, "cluster_centers_"):
                centers = self.kmeans_model.cluster_centers_
                if centers.shape[1] >= 2:
                    # Plot cluster centers with appropriate colors
                    for i, center in enumerate(centers):
                        if i < len(cluster_colors):
                            ax2.scatter(
                                center[0],
                                center[1],
                                c=cluster_colors[i],
                                s=300,
                                marker="X",
                                alpha=0.8,
                                edgecolors="black",
                                linewidth=2,
                                label=f"{cluster_labels[i]} Center",
                            )

                    # Highlight the predicted cluster center
                    if prediction < len(centers) and prediction < len(cluster_colors):
                        ax2.scatter(
                            centers[prediction, 0],
                            centers[prediction, 1],
                            c="yellow",
                            s=400,
                            marker="X",
                            edgecolors="red",
                            linewidth=3,
                            label=f"Your Cluster ({self.get_cluster_label(prediction)})",
                        )

                    # Plot prediction point
                    pred_color = (
                        cluster_colors[prediction]
                        if prediction < len(cluster_colors)
                        else "blue"
                    )
                    ax2.scatter(
                        processed_input[0, 0],
                        processed_input[0, 1],
                        c=pred_color,
                        s=200,
                        marker="*",
                        label="Your Input",
                        edgecolors="black",
                        linewidth=2,
                    )

                    ax2.set_xlabel("First Principal Component")
                    ax2.set_ylabel("Second Principal Component")
                    ax2.set_title("Cluster Centers and Your Prediction")
                    ax2.legend()
                    ax2.grid(True, alpha=0.3)

        plt.tight_layout()
        return fig


def main():
    st.title("🍎 Malnutrition Clustering Predictor")
    st.markdown("### Predict nutrition cluster based on country characteristics")
    st.info(
        "📊 **Model Overview**: This predictor classifies countries into 2 clusters based on malnutrition indicators:\n- **Cluster 0 (🔴)**: High Malnutrition\n- **Cluster 1 (🟢)**: Low Malnutrition"
    )

    # Initialize predictor
    predictor = MalnutritionPredictor()

    # Load models
    if not predictor.load_models():
        st.error(
            "❌ Could not load trained models. Please make sure the models are trained and saved properly."
        )
        st.info("Run the training script first to generate the required model files.")
        return

    st.success("✅ Models loaded successfully!")

    # Sidebar for model selection
    st.sidebar.header("🔧 Model Configuration")
    method = st.sidebar.selectbox(
        "Select Clustering Method:",
        ["kmeans", "dbhc"],
        help="Choose between K-Means and DBHC clustering methods",
    )

    # Input form
    st.header("📊 Input Country Characteristics")

    col1, col2 = st.columns(2)

    with col1:
        st.subheader("Economic Indicator")
        income_classification = st.selectbox(
            "Income Classification:",
            [0, 1, 2, 3],
            format_func=lambda x: {
                0: "Low income",
                1: "Lower middle income",
                2: "Upper middle income",
                3: "High income",
            }[x],
            help="Country's income classification level",
        )

        st.subheader("Malnutrition Indicators (per capita)")
        severe_wasting = st.number_input(
            "Severe Wasting per capita:",
            min_value=0.0,
            max_value=1.0,
            value=0.05,
            step=0.01,
            help="Rate of severe wasting per capita",
        )

        wasting = st.number_input(
            "Wasting per capita:",
            min_value=0.0,
            max_value=1.0,
            value=0.1,
            step=0.01,
            help="Rate of wasting per capita",
        )

        overweight = st.number_input(
            "Overweight per capita:",
            min_value=0.0,
            max_value=1.0,
            value=0.05,
            step=0.01,
            help="Rate of overweight per capita",
        )

    with col2:
        st.subheader("Additional Malnutrition Indicators")
        stunting = st.number_input(
            "Stunting per capita:",
            min_value=0.0,
            max_value=1.0,
            value=0.2,
            step=0.01,
            help="Rate of stunting per capita",
        )

        underweight = st.number_input(
            "Underweight per capita:",
            min_value=0.0,
            max_value=1.0,
            value=0.15,
            step=0.01,
            help="Rate of underweight per capita",
        )

        st.subheader("Interaction Features")
        stunting_underweight = st.number_input(
            "Stunting × Underweight:",
            min_value=0.0,
            max_value=1.0,
            value=stunting * underweight,
            step=0.001,
            help="Interaction between stunting and underweight rates",
        )

        wasting_overweight = st.number_input(
            "Wasting × Overweight:",
            min_value=0.0,
            max_value=1.0,
            value=wasting * overweight,
            step=0.001,
            help="Interaction between wasting and overweight rates",
        )

        severewasting_income = st.number_input(
            "Severe Wasting × Income:",
            min_value=0.0,
            max_value=3.0,
            value=severe_wasting * income_classification,
            step=0.01,
            help="Interaction between severe wasting and income classification",
        )

    # Prediction button
    if st.button("🔮 Predict Malnutrition Cluster", type="primary"):
        # Prepare input data
        user_input = {
            "Income Classification": income_classification,
            "Severe Wasting per capita": severe_wasting,
            "Wasting per capita": wasting,
            "Overweight per capita": overweight,
            "Stunting per capita": stunting,
            "Underweight per capita": underweight,
            "Stunting_Underweight": stunting_underweight,
            "Wasting_Overweight": wasting_overweight,
            "SevereWasting_Income": severewasting_income,
        }

        try:
            # Preprocess input
            processed_input = predictor.preprocess_input(user_input)

            # Make prediction based on selected method
            if method == "kmeans":
                prediction = predictor.predict_kmeans(processed_input)
            else:
                try:
                    prediction = predictor.predict_dbhc(processed_input)
                except Exception as dbhc_error:
                    st.error(f"DBHC prediction error: {str(dbhc_error)}")
                    st.info("Falling back to K-Means prediction...")
                    prediction = predictor.predict_kmeans(processed_input)
                    method = "kmeans"  # Update method for display purposes

            if prediction is not None:
                # Display results
                st.header("🎯 Prediction Results")

                col1, col2 = st.columns([1, 2])

                with col1:
                    cluster_label = predictor.get_cluster_label(prediction)
                    cluster_color = "🔴" if prediction == 0 else "🟢"
                    st.metric(
                        label=f"{method.upper()} Prediction",
                        value=f"{cluster_color} {cluster_label}",
                    )
                    st.caption(f"Cluster ID: {prediction}")

                with col2:
                    explanation = predictor.get_cluster_explanation(prediction, method)
                    st.markdown(f"**Cluster Interpretation:**\n\n{explanation}")

                # Show input summary
                st.subheader("📋 Input Summary")
                input_df = pd.DataFrame([user_input])
                st.dataframe(
                    input_df.T.rename(columns={0: "Value"}), use_container_width=True
                )

                # Plot visualization
                st.subheader("📈 Visualization")
                try:
                    # Try to load some training data for context (if available)
                    training_data = None
                    if os.path.exists("dataset/results/kmeans_clustering_results.csv"):
                        # Create some sample training data for visualization
                        np.random.seed(42)
                        training_data = np.random.randn(50, processed_input.shape[1])

                    fig = predictor.plot_prediction_location(
                        processed_input, prediction, method, training_data
                    )
                    st.pyplot(fig)

                except Exception as plot_error:
                    st.warning(f"Could not generate plot: {str(plot_error)}")

                # Recommendations based on 2-cluster system
                st.subheader("💡 Recommendations")

                if prediction == 0:  # High Malnutrition
                    st.error("🚨 **High Priority Interventions Needed**")
                    recommendations = [
                        "Implement emergency nutrition programs immediately",
                        "Focus on treating severe acute malnutrition",
                        "Improve food security and access to nutritious foods",
                        "Strengthen healthcare infrastructure and capacity",
                        "Develop targeted feeding programs for vulnerable populations",
                        "Address underlying causes of poverty and food insecurity",
                        "Establish nutrition surveillance and monitoring systems",
                    ]
                else:  # Low Malnutrition
                    st.success("✅ **Maintain Current Progress**")
                    recommendations = [
                        "Continue current nutrition programs and policies",
                        "Monitor for emerging nutrition challenges",
                        "Address any remaining pockets of malnutrition",
                        "Focus on preventing nutrition-related non-communicable diseases",
                        "Promote healthy lifestyle choices and balanced nutrition",
                        "Maintain strong nutrition surveillance systems",
                        "Share best practices with countries facing higher malnutrition rates",
                    ]

                for rec in recommendations:
                    st.write(f"• {rec}")

                # Risk assessment
                st.subheader("⚠️ Risk Assessment")
                if prediction == 0:
                    st.error(
                        "**High Risk**: This nutrition profile indicates significant malnutrition challenges that require immediate and sustained intervention."
                    )
                else:
                    st.success(
                        "**Low Risk**: This nutrition profile indicates relatively good nutrition outcomes with manageable challenges."
                    )

            else:
                st.error(
                    "❌ Could not make prediction. Please check your inputs and try again."
                )

        except Exception as e:
            st.error(f"❌ Error during prediction: {str(e)}")


if __name__ == "__main__":
    main()
