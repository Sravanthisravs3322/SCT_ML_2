# ===== Memory Leak Fix (Windows + MKL) ===== #
import os
os.environ["OMP_NUM_THREADS"] = "1"  # Avoid memory leaks with MKL
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"

# ===== Importing Necessary Libraries ===== #
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

# ===== Load the Dataset ===== #
# Replace with the path to your dataset
file_path = r"C:\Users\Sravanti\Downloads\Mall_Customers.csv"
data = pd.read_csv(file_path)

# Display the first few rows
print("First few rows of the dataset:")
print(data.head())

# ===== Data Preprocessing ===== #
# Checking for missing values
print("\nMissing values in dataset:")
print(data.isnull().sum())

# Select relevant features (e.g., Annual Income and Spending Score)
features = data[['Annual Income (k$)', 'Spending Score (1-100)']]

# Scaling the data for better clustering performance
scaler = StandardScaler()
scaled_features = scaler.fit_transform(features)

# ===== K-means Clustering ===== #
# Elbow Method to determine the optimal number of clusters
inertia = []
k_values = range(1, 11)
for k in k_values:
    kmeans = KMeans(n_clusters=k, n_init=10, random_state=42)  # Explicitly setting n_init to suppress warning
    kmeans.fit(scaled_features)
    inertia.append(kmeans.inertia_)

# Plot the Elbow Curve
plt.figure(figsize=(8, 6))
plt.plot(k_values, inertia, marker='o', linestyle='--')
plt.title("Elbow Method for Optimal K")
plt.xlabel("Number of Clusters (k)")
plt.ylabel("Inertia")
plt.show()

# Based on the elbow curve, choose optimal k (e.g., 5)
optimal_k = 5
kmeans = KMeans(n_clusters=optimal_k, n_init=10, random_state=42)  # Explicitly setting n_init
data['Cluster'] = kmeans.fit_predict(scaled_features)

# ===== Visualization ===== #
# Visualizing the clusters
plt.figure(figsize=(10, 8))
sns.scatterplot(
    x=data['Annual Income (k$)'],
    y=data['Spending Score (1-100)'],
    hue=data['Cluster'],
    palette='viridis',
    s=100
)
plt.title("Customer Segments")
plt.xlabel("Annual Income (k$)")
plt.ylabel("Spending Score (1-100)")
plt.legend(title="Cluster")
plt.show()

# ===== Insights ===== #
# Display the cluster centroids
centroids = scaler.inverse_transform(kmeans.cluster_centers_)
centroid_df = pd.DataFrame(centroids, columns=['Annual Income (k$)', 'Spending Score (1-100)'])
print("\nCluster Centroids:")
print(centroid_df)

# Save the clustered dataset to a new CSV
output_path = r"C:\Users\Sravanti\Documents\Clustered_Customers.csv"  # Replace with your desired save location
data.to_csv(output_path, index=False)
print(f"\nClustered customer data saved as '{output_path}'")
