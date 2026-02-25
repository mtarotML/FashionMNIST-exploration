import streamlit as st
import numpy as np
from sklearn.cluster import KMeans
import gc
import psutil
import os
import plotly.express as px
import pandas as pd
pd.options.plotting.backend = "plotly"


st.set_page_config(page_title="FashionMNIST Viewer", 
                   layout="wide",
                   page_icon = "logo.png")

# st.sidebar.write(psutil.Process(os.getpid()).memory_info().rss / (1024 * 1024))

@st.cache_data(show_spinner="Loading data...")
def load_local_data():
    try:
        x_train = np.load('f_mnist_x_train.npy')
        y_train = np.load('f_mnist_y_train.npy')     
        return x_train, y_train
    except FileNotFoundError:
        print("Error: Local files not found.")


X_train, y_train= load_local_data()[0].reshape(-1,784),load_local_data()[1]

f_mnist_labels = {
    0: "T-shirt/top",
    1: "Trouser",
    2: "Pullover",
    3: "Dress",
    4: "Coat",
    5: "Sandal",
    6: "Shirt",
    7: "Sneaker",
    8: "Bag",
    9: "Ankle boot"
}



@st.cache_data()
def train_model_clustering(X_train):
    model = KMeans(10)
    pred = model.fit_predict(X_train)
    return pred

pred = train_model_clustering(X_train)

clusters = pd.DataFrame({"label" : y_train,
                         "cluster" : pred})

clusters["label"] = clusters["label"].apply(lambda x : f_mnist_labels[x])

repartition = clusters.groupby("label")["cluster"].value_counts().reset_index()


###display

st.title("Fashion-MNIST clustering")

st.markdown("A simple tool to visualize clustering performance on Fashion-MNIST dataset.")

st.write(repartition.plot.bar(x = "cluster" ,y = "count",color = "label"))

st.info("Data loaded and normalized. Ready for further processing.")