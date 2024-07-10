import pandas as pd
import numpy as np

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

def hotdeck_random(df):
    imputed = df.copy()
    for column in imputed.columns:
        missing = imputed[column].isnull()
        num_missing = missing.sum()
        if num_missing > 0:
            sampled_values = imputed.loc[~missing, column].sample(num_missing, replace=True).values
            imputed.loc[missing, column] = sampled_values
    return imputed

hotdeck_imputed_random = hotdeck_random(data)
print("\nHot Deck Imputation (Random):")
print(hotdeck_imputed_random)
