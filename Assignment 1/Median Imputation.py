import numpy as np
import pandas as pd

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

median_imputed = data.copy()
median_imputed['A'] = median_imputed['A'].fillna(median_imputed['A'].median())
median_imputed['B'] = median_imputed['B'].fillna(median_imputed['B'].median())
median_imputed['C'] = median_imputed['C'].fillna(median_imputed['C'].median())
print("\nMedian Imputation:")
print(median_imputed)
