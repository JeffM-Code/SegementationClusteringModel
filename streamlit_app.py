import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score, davies_bouldin_score
from sklearn.decomposition import PCA
import pickle

file_path = 'test.csv'
data = pd.read_csv(file_path, delimiter=',')
data.replace('?', np.nan, inplace=True)
data['Date'] = data['Date'].astype(str)
data['Time'] = data['Time'].astype(str)
data['Datetime'] = pd.to_datetime(data['Date'] + ' ' + data['Time'], dayfirst=True, errors='coerce')
data = data.drop(columns=['Date', 'Time'])
data = data.dropna(subset=['Datetime'])

for column in data.columns:
    data[column] = pd.to_numeric(data[column], errors='coerce')

data = data.fillna(data.mean(numeric_only=True))

st.title('Segmentation Clustering Model')

data_percentage = st.slider('Percentage of data to use for clustering', 10, 100, 50)
plot_dimension = st.slider('Number of dimensions for PCA plot', 2, 3, 2)

if st.button('Predict'):
    scaler = StandardScaler()
    scaled_data = scaler.fit_transform(data.drop(columns=['Datetime']))
    scaled_data = pd.DataFrame(scaled_data, columns=data.columns[:-1])

    sampled_data = scaled_data.sample(frac=data_percentage / 100, random_state=42)

    k = 3
    kmeans = KMeans(n_clusters=k, random_state=42)
    sampled_data['Cluster'] = kmeans.fit_predict(sampled_data)

    silhouette_avg = silhouette_score(sampled_data, sampled_data['Cluster'])
    db_index = davies_bouldin_score(sampled_data, sampled_data['Cluster'])

    st.write(f'Silhouette Score: {silhouette_avg}')
    st.write(f'Davies-Bouldin Index: {db_index}')

    pca = PCA(n_components=plot_dimension)
    pca_data = pca.fit_transform(sampled_data.drop(columns=['Cluster']))

    fig = plt.figure()
    
    if plot_dimension == 2:
        ax = fig.add_subplot(111)
        scatter = ax.scatter(pca_data[:, 0], pca_data[:, 1], c=sampled_data['Cluster'], cmap='viridis', marker='.')
        legend1 = ax.legend(*scatter.legend_elements(), title="Clusters")
        ax.add_artist(legend1)
        plt.title('Clustering (2D)')
        plt.xlabel('PCA Component 1')
        plt.ylabel('PCA Component 2')
    elif plot_dimension == 3:
        from mpl_toolkits.mplot3d import Axes3D
        ax = fig.add_subplot(111, projection='3d')
        scatter = ax.scatter(pca_data[:, 0], pca_data[:, 1], pca_data[:, 2], c=sampled_data['Cluster'], cmap='viridis', marker='.')
        legend1 = ax.legend(*scatter.legend_elements(), title="Clusters")
        ax.add_artist(legend1)
        ax.set_title('Clustering (3D)')
        ax.set_xlabel('PCA Component 1')
        ax.set_ylabel('PCA Component 2')
        ax.set_zlabel('PCA Component 3')
    
    st.pyplot(fig)

    model_file_path = 'kmeans_model.pkl'
    scaler_file_path = 'scaler.pkl'

    with open(model_file_path, 'wb') as model_file:
        pickle.dump(kmeans, model_file)

    with open(scaler_file_path, 'wb') as scaler_file:
        pickle.dump(scaler, scaler_file)
