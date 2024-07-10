import numpy as np
import pandas as pd
from sklearn.experimental import enable_hist_gradient_boosting
from sklearn.ensemble import HistGradientBoostingRegressor

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

def regression_impute(df, target_column, predictor_columns):
    imputed = df.copy()
    known = imputed[imputed[target_column].notnull()]
    unknown = imputed[imputed[target_column].isnull()]
    
    if len(unknown) > 0:
        X_train = known[predictor_columns]
        y_train = known[target_column]
        X_test = unknown[predictor_columns]
        
        model = HistGradientBoostingRegressor()
        model.fit(X_train, y_train)
        predicted = model.predict(X_test)
        imputed.loc[imputed[target_column].isnull(), target_column] = predicted
    
    return imputed

# Impute 'A' using 'B' and 'C'
data_imputed_A = regression_impute(data, 'A', ['B', 'C'])

# Impute 'B' using 'A' and 'C'
data_imputed_B = regression_impute(data_imputed_A, 'B', ['A', 'C'])

data_imputed_C = regression_impute(data_imputed_B, 'C', ['A', 'B'])

print("\nRegression Imputation using HistGradientBoostingRegressor:")
print(data_imputed_C)
