import streamlit as st
import numpy as np
import plotly.express as px
from sklearn.cluster import MiniBatchKMeans
from sklearn.datasets import fetch_openml
import gc


# --- PAGE CONFIG ---
st.set_page_config(
    page_title="FashionMNIST Clustering Analysis",
    layout="wide"
    #page_icon= "logo.png"
)

# -----------------------------
# DATA LOADING (LOW MEMORY)
# -----------------------------
@st.cache_data(show_spinner=False)
def load_subset(max_samples):
    # Chargement brut
    dataset = fetch_openml(
        "Fashion-MNIST",
        version=1,
        as_frame=False,
        parser="auto",
    )

    # Sous-échantillonnage IMMÉDIAT pour éviter garder 70k en RAM
    total_samples = dataset.data.shape[0]
    indices = np.random.choice(total_samples, max_samples, replace=False)

    # Conversion directe en float32 (2x moins de RAM que float64)
    images = dataset.data[indices].astype(np.float32) / 255.0
    labels = dataset.target[indices]

    # Libération mémoire
    del dataset
    gc.collect()

    return images, labels


# --- HEADER ---
st.title("FashionMNIST Unsupervised Exploration")

st.markdown("""
Clustering non supervisé avec optimisation mémoire.
""")

# --- SIDEBAR ---
st.sidebar.header("Model Parameters")

n_clusters = st.sidebar.slider("Number of Clusters (K)", 2, 20, 10)

subset_size = st.sidebar.select_slider(
    "Sample Size (RAM optimized)",
    options=[1000, 2000, 5000, 10000],
    value=2000
)

run_kmeans = st.sidebar.button("Run Clustering", type="primary")

# --- LOAD DATA ---
with st.spinner("Loading optimized subset..."):
    X_subset, y_subset = load_subset(subset_size)

st.subheader("Dataset Overview")
st.write(f"Sample size: {subset_size}")

# -----------------------------
# CLUSTERING
# -----------------------------
if run_kmeans:
    st.divider()

    with st.spinner("Training MiniBatchKMeans..."):
        # MiniBatchKMeans consomme beaucoup moins de RAM
        kmeans = MiniBatchKMeans(
            n_clusters=n_clusters,
            batch_size=256,
            random_state=42,
            n_init=5,
        )
        clusters = kmeans.fit_predict(X_subset)

    # Comptage sans DataFrame intermédiaire lourd
    labels_unique = np.unique(y_subset)
    cluster_ids = np.arange(n_clusters)

    counts = []
    for label in labels_unique:
        for cluster in cluster_ids:
            mask = (y_subset == label) & (clusters == cluster)
            counts.append({
                "label": label,
                "cluster": f"Cluster {cluster}",
                "count": int(np.sum(mask))
            })

    # Plot
    fig = px.bar(
        counts,
        x="cluster",
        y="count",
        color="label",
        barmode="stack",
        title=f"Distribution across {n_clusters} Clusters",
    )

    st.plotly_chart(fig, use_container_width=True)

    # Nettoyage mémoire
    del clusters
    gc.collect()

else:
    st.warning("Adjust parameters and click 'Run Clustering'")

st.markdown("---")
