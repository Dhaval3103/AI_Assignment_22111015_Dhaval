import numpy as np
import pandas as pd
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import IterativeImputer

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

iterative_imputer = IterativeImputer(max_iter=10, random_state=0)
multiple_imputed = pd.DataFrame(iterative_imputer.fit_transform(data), columns=data.columns)
print("\nMultiple Imputation (Iterative):")
print(multiple_imputed)
