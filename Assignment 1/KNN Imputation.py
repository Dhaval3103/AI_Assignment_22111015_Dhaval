import numpy as np
import pandas as pd
from sklearn.impute import KNNImputer

# Generate example data
np.random.seed(0)
data = pd.DataFrame({
    'A': np.random.rand(10),
    'B': np.random.rand(10),
    'C': np.random.randint(1, 4, size=10)
})
data.loc[[1, 4, 7], 'A'] = np.nan  # Introduce missing values
data.loc[[2, 5], 'B'] = np.nan
data.loc[[0, 3, 8], 'C'] = np.nan

print("Original Data:")
print(data)


knn_imputer = KNNImputer(n_neighbors=2)
knn_imputed = pd.DataFrame(knn_imputer.fit_transform(data), columns=data.columns)
print("\nKNN Imputation:")
print(knn_imputed)

