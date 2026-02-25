import streamlit as st
import numpy as np
from sklearn.datasets import fetch_openml
import gc

# --- PAGE CONFIG ---
st.set_page_config(page_title="FashionMNIST Viewer", layout="wide")

# -----------------------------
# DATA LOADING
# -----------------------------
@st.cache_data(show_spinner="Fetching data from OpenML...")
def load_fashion_mnist(n_samples=50):
    # Fetch data
    dataset = fetch_openml("Fashion-MNIST", version=1, as_frame=False, parser="auto")
    
    # Pick random indices
    indices = np.random.choice(dataset.data.shape[0], n_samples, replace=False)
    
    # Process images: (Samples, 784) -> (Samples, 28, 28)
    images = dataset.data[indices].reshape(-1, 28, 28) / 255.0
    labels = dataset.target[indices]
    
    del dataset
    gc.collect()
    return images, labels

# --- UI ---
st.title("Fashion-MNIST Data Fetcher")
st.markdown("A simple tool to pull random samples from the Fashion-MNIST dataset.")

# Sidebar controls
num_to_show = st.sidebar.slider("Number of images to fetch", 5, 100, 20)

# Load data
images, labels = load_fashion_mnist(num_to_show)

# --- DISPLAY GRID ---
st.subheader(f"Displaying {num_to_show} random items")

# Define Fashion MNIST label map for readability
label_map = {
    '0': 'T-shirt/top', '1': 'Trouser', '2': 'Pullover', '3': 'Dress', '4': 'Coat',
    '5': 'Sandal', '6': 'Shirt', '7': 'Sneaker', '8': 'Bag', '9': 'Ankle boot'
}

# Create a grid using columns
cols = st.columns(5) 
for i, (img, lbl) in enumerate(zip(images, labels)):
    with cols[i % 5]:
        # Use st.image with the numpy array
        st.image(img, caption=f"{label_map[lbl]} (ID:{lbl})", use_container_width=True)

st.divider()
st.info("Data loaded and normalized. Ready for further processing.")