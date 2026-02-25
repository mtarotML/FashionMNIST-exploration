import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import torchvision
from sklearn.cluster import KMeans
from sklearn.model_selection import train_test_split

# --- PAGE CONFIG ---
st.set_page_config(page_title="FashionMNIST Clustering Analysis", layout="wide")

@st.cache_data
def load_and_prep_data():
    # Load dataset
    dataset = torchvision.datasets.FashionMNIST(root="./data", train=True, download=True)
    images = dataset.data.numpy().reshape(-1, 784).astype(float)
    labels = np.array([dataset.classes[t] for t in dataset.targets.numpy()])
    
    df = pd.DataFrame({"label": labels})
    df['image'] = list(images)
    
    # Split data (matching notebook logic)
    X_train, X_test, y_train, y_test = train_test_split(
        df["image"], df["label"], test_size=0.2, random_state=42
    )
    
    df_train = pd.DataFrame({"image": X_train, "label": y_train})
    return df_train, X_train

# --- HEADER ---
st.title("FashionMNIST Unsupervised Exploration")
st.markdown("""
This application demonstrates how **K-Means Clustering** groups different apparel items from the FashionMNIST dataset. 
Even without labels, can the algorithm distinguish a 'Sandal' from a 'Coat'?
""")

# --- LOAD DATA ---
with st.spinner("Loading FashionMNIST data..."):
    df_train, X_train = load_and_prep_data()

# --- SIDEBAR / CONTROLS ---
st.sidebar.header("Model Parameters")
n_clusters = st.sidebar.slider("Number of Clusters (K)", 2, 20, 10)
run_kmeans = st.sidebar.button("Run Clustering")


st.subheader("Dataset Overview")
st.write(f"Total training samples: **{len(df_train)}**")


# --- CLUSTERING LOGIC ---
if run_kmeans:
    st.divider()
    st.subheader("K-Means Results")
    
    with st.spinner("Training K-Means..."):
        # We use a subset or cached version for speed in the UI if needed
        kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
        predictions = kmeans.fit_predict(np.stack(df_train["image"].values))
        df_train["cluster"] = predictions

    # Calculate repartition
    repartition = df_train.groupby(["label", "cluster"]).size().reset_index(name="count")

    # Plotly Chart
    fig = px.bar(
        repartition, 
        x="cluster", 
        y="count", 
        color="label",
        title=f"Distribution of Fashion Items across {n_clusters} Clusters",
        barmode="stack"
    )
    
    st.plotly_chart(fig, use_container_width=True)

    # Analysis Note
    st.info("""
    **How to read this:** Each bar represents a Cluster. The colors show which actual labels fell into that cluster. 
    A 'pure' cluster (mostly one color) means the algorithm successfully identified unique features for that item type.
    """)
else:
    st.warning("👈 Adjust parameters and click 'Run Clustering' in the sidebar to see the results!")

# --- FOOTER ---
st.markdown("---")