# Importing necessary libraries
import numpy as np
from sklearn.decomposition import PCA

# Sample data: 2D points
X = np.array([[1, 2], [1, 4], [1, 0],
              [4, 2], [4, 4], [4, 0]])

# Applying PCA
pca = PCA(n_components=2)
X_pca = pca.fit_transform(X)

# Printing the results
print("Original Data:")
print(X)
print("\nTransformed Data:")
print(X_pca)
print("\nExplained Variance Ratio:")
print(pca.explained_variance_ratio_)
print("\nPrincipal Components:")
print(pca.components_)
