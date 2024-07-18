# Importing necessary libraries
import numpy as np
from sklearn.cluster import KMeans

# Sample data: 2D points
X = np.array([[1, 2], [1, 4], [1, 0],
              [4, 2], [4, 4], [4, 0]])

# Applying KMeans algorithm
kmeans = KMeans(n_clusters=2, random_state=0).fit(X)

# Getting cluster centers and labels
centers = kmeans.cluster_centers_
labels = kmeans.labels_

# Printing the results
print("Cluster Centers:")
print(centers)
print("\nCluster Labels for each point:")
for point, label in zip(X, labels):
    print(f"Point {point} is in cluster {label}")
