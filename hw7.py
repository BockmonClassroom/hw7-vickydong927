import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import pandas as pd
import numpy as np
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans, AgglomerativeClustering
from scipy.cluster.hierarchy import dendrogram, linkage
from scipy.spatial.distance import pdist


# Part 1: 3D K-means Clustering
# Load the dataset
data = pd.read_csv("Data\Spotify_YouTube.csv")
# Select the three features we want to cluster
X = data[["Liveness", "Energy", "Loudness"]]

# Standardize the data (important for features on different scales)
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Generate elbow plot to determine optimal number of clusters
sse = []  # Sum of squared errors (inertia)
max_k = 10  # Maximum number of clusters to test
for k in range(1, max_k+1):
    kmeans = KMeans(n_clusters=k, n_init=10, random_state=42)  # Run K-means
    kmeans.fit(X_scaled)  # Fit to our scaled data
    sse.append(kmeans.inertia_)  # Store the inertia (SSE)

# Plot the elbow curve
plt.figure(figsize=(8, 5))
plt.plot(range(1, max_k+1), sse, marker='o', linestyle='--')
plt.xlabel('Number of Clusters (K)')
plt.ylabel('Sum of Squared Distances (SSE)')
plt.title('Elbow Method for Optimal K')
plt.grid(True)
plt.show()

# Based on the elbow plot, select optimal K (here we use K=3 as example)
optimal_k = 3
# Initialize and fit K-means with our chosen K
kmeans = KMeans(n_clusters=optimal_k, n_init=10, random_state=42)
y_km = kmeans.fit_predict(X_scaled)  # Get cluster assignments

# Get cluster centers and transform back to original scale for interpretation
cluster_centers_scaled = kmeans.cluster_centers_
cluster_centers_original = scaler.inverse_transform(cluster_centers_scaled)

# Create 3D visualization
fig = plt.figure(figsize=(12, 8))
ax = fig.add_subplot(111, projection='3d')

# Define colors and markers for different clusters
colors = ['r', 'g', 'b', 'c', 'm', 'y', 'k']  # Red, Green, Blue, etc.
markers = ['o', 's', '^', 'D', 'P', 'X', '*']  # Circle, Square, Triangle, etc.

# Plot data points for each cluster
for i in range(optimal_k):
    # Get data points belonging to current cluster
    cluster_data = X[y_km == i]
    ax.scatter(cluster_data["Liveness"],
               cluster_data["Energy"],
               cluster_data["Loudness"],
               s=40,  # Point size
               c=colors[i],  # Color
               marker=markers[i%7],  # Marker style (cycling through list)
               label=f'Cluster {i+1}')  # Legend label

# Plot cluster centers (in original scale)
ax.scatter(cluster_centers_original[:, 0],
           cluster_centers_original[:, 1],
           cluster_centers_original[:, 2],
           s=300,  # Larger size for centroids
           c='black',  # Black color
           marker='*',  # Star marker
           label='Centroids')  # Legend label

# Add axis labels and title
ax.set_xlabel('Liveness', fontsize=12)
ax.set_ylabel('Energy', fontsize=12)
ax.set_zlabel('Loudness (dB)', fontsize=12)
plt.title(f'3D Cluster Visualization (K={optimal_k})', fontsize=14)
ax.legend(loc='upper right', bbox_to_anchor=(1.1, 1))  # Position legend

# Adjust viewing angle (elevation and azimuth)
ax.view_init(elev=20, azim=45)

# Ensure proper layout and display the plot
plt.tight_layout()
plt.show()


# Part 2: Hierarchical Clustering for Individual Columns
def plot_dendrogram(feature_data, feature_name, method='ward'):
    plt.figure(figsize=(10, 6))

    # Calculate distances and perform hierarchical clustering
    distance_matrix = pdist(feature_data.reshape(-1, 1))
    Z = linkage(distance_matrix, method=method)

    # Plot dendrogram
    dendrogram(Z, orientation='top',
               labels=np.arange(len(feature_data)),
               distance_sort='descending',
               show_leaf_counts=True)

    plt.title(f'Hierarchical Clustering Dendrogram ({feature_name})')
    plt.xlabel('Sample Index')
    plt.ylabel('Distance')
    plt.show()


def analyze_feature_clusters(data, feature_name, n_clusters=2):
    feature_data = data[feature_name].values
    scaled_data = StandardScaler().fit_transform(feature_data.reshape(-1, 1))

    # Hierarchical clustering
    hc = AgglomerativeClustering(n_clusters=n_clusters,
                                 affinity='euclidean',
                                 linkage='ward')
    clusters = hc.fit_predict(scaled_data)

    # Plot distribution with clusters
    plt.figure(figsize=(10, 6))
    for cluster in range(n_clusters):
        cluster_data = feature_data[clusters == cluster]
        plt.hist(cluster_data, bins=30, alpha=0.5,
                 label=f'Cluster {cluster + 1}')

    plt.title(f'{feature_name} Distribution by Cluster')
    plt.xlabel(feature_name)
    plt.ylabel('Frequency')
    plt.legend()
    plt.show()

    # Print cluster characteristics
    print(f"\nCluster Characteristics for {feature_name}:")
    for cluster in range(n_clusters):
        cluster_values = feature_data[clusters == cluster]
        print(f"Cluster {cluster + 1}:")
        print(f"  Count: {len(cluster_values)}")
        print(f"  Mean: {cluster_values.mean():.2f}")
        print(f"  Range: {cluster_values.min():.2f} - {cluster_values.max():.2f}")
        print(f"  Std Dev: {cluster_values.std():.2f}\n")


# Analyze each feature
features_to_analyze = ['Liveness', 'Energy', 'Loudness']

for feature in features_to_analyze:
    # Get feature data
    feature_data = data[feature].values.reshape(-1, 1)

    # Plot dendrogram
    plot_dendrogram(feature_data, feature)

    # Analyze clusters (assuming 2 clusters based on dendrogram)
    analyze_feature_clusters(data, feature, n_clusters=3)