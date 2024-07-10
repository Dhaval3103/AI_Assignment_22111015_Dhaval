import numpy as np
import pandas as pd

# Generating example data
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

#Mean imputation
mean_imputed = data.copy()
mean_imputed['A'] = mean_imputed['A'].fillna(mean_imputed['A'].mean())
mean_imputed['B'] = mean_imputed['B'].fillna(mean_imputed['B'].mean())
print("\nMean Imputation:")
print(mean_imputed)
