import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
from sklearn.cluster import KMeans
from sklearn.model_selection import train_test_split
from sklearn.datasets import fetch_openml

# --- PAGE CONFIG ---
st.set_page_config(page_title="FashionMNIST Clustering Analysis", layout="wide")

@st.cache_data
def load_and_prep_data():
    # fetch_openml handles the download and caching locally
    # parser='auto' is correct for newer versions of sklearn
    dataset = fetch_openml('Fashion-MNIST', version=1, as_frame=False, parser='auto')
    
    # images are already flattened (784 columns) if as_frame=False
    images = dataset.data.astype(float)
    labels = dataset.target.astype(str) # Labels come as strings/categories
    
    return images, labels

# --- HEADER ---
st.title("👕 FashionMNIST Unsupervised Exploration")
st.markdown("""
This application demonstrates how **K-Means Clustering** groups different apparel items. 
Even without labels, can the algorithm distinguish a 'Sandal' from a 'Coat'?
""")

# --- LOAD DATA ---
with st.spinner("Fetching FashionMNIST from OpenML..."):
    images, labels = load_and_prep_data()

# --- SIDEBAR / CONTROLS ---
st.sidebar.header("Model Parameters")
n_clusters = st.sidebar.slider("Number of Clusters (K)", 2, 20, 10)
subset_size = st.sidebar.select_slider(
    "Sample Size (for speed)", 
    options=[1000, 5000, 10000, 20000], 
    value=5000
)
run_kmeans = st.sidebar.button("Run Clustering", type="primary")

# --- DATA PREP ---
# We take a random subset to keep the UI snappy
indices = np.random.choice(len(images), subset_size, replace=False)
X_subset = images[indices]
y_subset = labels[indices]

st.subheader("Dataset Overview")
st.write(f"Analyzing a random sample of **{subset_size}** items.")



# --- CLUSTERING LOGIC ---
if run_kmeans:
    st.divider()
    
    with st.spinner(f"Training K-Means on {subset_size} samples..."):
        # n_init='auto' or 10 is good for stability
        kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
        clusters = kmeans.fit_predict(X_subset)
    
    # Create results dataframe
    results_df = pd.DataFrame({
        "label": y_subset,
        "cluster": [f"Cluster {c}" for c in clusters]
    })

    # Grouping for the chart
    repartition = results_df.groupby(["label", "cluster"]).size().reset_index(name="count")

    # Plotly Chart
    fig = px.bar(
        repartition, 
        x="cluster", 
        y="count", 
        color="label",
        title=f"Distribution of Fashion Items across {n_clusters} Clusters",
        barmode="stack",
        category_orders={"cluster": [f"Cluster {i}" for i in range(n_clusters)]}
    )
    
    st.plotly_chart(fig, use_container_width=True)

    # Analysis Note
    st.info("""
    **How to read this:** Each bar is a Cluster. The colors show the actual labels. 
    A 'pure' cluster (mostly one color) indicates the algorithm found features unique to that item type.
    """)
else:
    st.warning("👈 Adjust parameters and click 'Run Clustering' in the sidebar to begin!")

# --- FOOTER ---
st.markdown("---")