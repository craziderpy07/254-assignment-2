
import numpy as np
from sklearn.datasets import load_iris
from sklearn.cluster import KMeans
from sklearn.metrics import accuracy_score, confusion_matrix
from scipy.optimize import linear_sum_assignment

# Load Iris dataset
iris = load_iris()
X = iris.data
y = iris.target

# Perform K-Means clustering with K=3
kmeans = KMeans(n_clusters=3, random_state=42, n_init=10)
clusters = kmeans.fit_predict(X)

# Get cluster centroids
centroids = kmeans.cluster_centers_

# Calculate RMSE for each cluster
print("K-Means Clustering Results")

for i in range(3):
    cluster_points = X[clusters == i]

    rmse = np.sqrt(
        np.mean(np.sum((cluster_points - centroids[i]) ** 2, axis=1))
    )

    print(f"Cluster {i}:")
    print(f"  Number of samples: {len(cluster_points)}")
    print(f"  RMSE: {rmse:.4f}")

# Compare clusters with actual Iris classes
print("\nCluster Quality Validation")

conf_matrix = confusion_matrix(y, clusters)
print("Confusion Matrix:")
print(conf_matrix)

# Match cluster IDs to actual class labels
row_ind, col_ind = linear_sum_assignment(-conf_matrix)

cluster_mapping = {
    cluster: actual_class
    for actual_class, cluster in zip(row_ind, col_ind)
}

predicted_labels = np.array([
    cluster_mapping[cluster] for cluster in clusters
])

# Calculate clustering accuracy
accuracy = accuracy_score(y, predicted_labels)

print(f"\nK-Means Clustering Accuracy: {accuracy * 100:.2f}%")

# Display comparison with actual classes
print("\nActual Iris Classes:")
for i, name in enumerate(iris.target_names):
    print(f"{i}: {name}")

print("\nCluster-to-Class Mapping:")
for cluster, actual_class in cluster_mapping.items():
    print(f"Cluster {cluster}: {iris.target_names[actual_class]}")
