import pickle
import numpy as np
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt

from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

# Load models
with open("train/kmeans_model.pkl", "rb") as f:
    kmeans_model = pickle.load(f)

with open("train/dbhc_model.pkl", "rb") as f:
    dbhc_model = pickle.load(f)


# Load and preprocess data
def load_data_for_scaler():
    df = pd.read_csv("dataset/country-wise-average.csv")
    features = [
        "Income Classification",
        "Severe Wasting",
        "Wasting",
        "Overweight",
        "Stunting",
        "Underweight",
        "U5 Population ('000s)",
    ]
    df[features] = df[features].fillna(df[features].mean())
    return df


# Fit scaler
df_full = load_data_for_scaler()
feature_names = [
    "Income Classification",
    "Severe Wasting",
    "Wasting",
    "Overweight",
    "Stunting",
    "Underweight",
    "U5 Population ('000s)",
]
X_full = df_full[feature_names]
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X_full)

# PCA for visualization
pca = PCA(n_components=2)
X_pca = pca.fit_transform(X_scaled)

# Cluster predictions
kmeans_labels = kmeans_model.predict(X_scaled)
dbhc_centroids = np.array([c["centroid"] for c in dbhc_model])
dbhc_labels = [np.argmin(np.linalg.norm(dbhc_centroids - x, axis=1)) for x in X_scaled]

# Streamlit UI
st.title("Clustering Negara Berdasarkan Indikator Gizi Anak")

st.markdown("Masukkan nilai-nilai indikator berikut:")

user_input = {}
for feature in feature_names:
    user_input[feature] = st.number_input(feature, format="%.3f")

method = st.selectbox("Pilih metode clustering:", ["KMeans", "DBHC"])

if st.button("Prediksi Cluster"):
    user_features = pd.DataFrame([user_input], columns=feature_names)
    user_scaled = scaler.transform(user_features)

    if method == "KMeans":
        cluster = kmeans_model.predict(user_scaled)[0]
        st.success(f"Negara Anda masuk ke dalam cluster KMeans: {cluster}")

        # Plot
        st.subheader("Visualisasi Cluster (KMeans)")
        fig, ax = plt.subplots()
        scatter = ax.scatter(X_pca[:, 0], X_pca[:, 1], c=kmeans_labels, cmap="Set2")
        ax.scatter(
            *pca.transform(user_scaled).T,
            color="red",
            marker="x",
            s=100,
            label="Input Anda",
        )
        ax.set_xlabel("PC1")
        ax.set_ylabel("PC2")
        ax.legend()
        st.pyplot(fig)

    else:
        centroids = np.array([c["centroid"] for c in dbhc_model])
        distances = np.linalg.norm(centroids - user_scaled, axis=1)
        cluster = np.argmin(distances)
        st.success(f"Negara Anda masuk ke dalam cluster DBHC: {cluster}")

        # Plot
        st.subheader("Visualisasi Cluster (DBHC)")
        fig, ax = plt.subplots()
        scatter = ax.scatter(X_pca[:, 0], X_pca[:, 1], c=dbhc_labels, cmap="Set3")
        ax.scatter(
            *pca.transform(user_scaled).T,
            color="red",
            marker="x",
            s=100,
            label="Input Anda",
        )
        ax.set_xlabel("PC1")
        ax.set_ylabel("PC2")
        ax.legend()
        st.pyplot(fig)
