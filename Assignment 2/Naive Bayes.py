from sklearn.naive_bayes import GaussianNB
import numpy as np

# Example data
X = np.array([[1, 1], [2, 2], [3, 3], [4, 4]])
y = np.array([0, 0, 1, 1])

# Create and train the model
model = GaussianNB()
model.fit(X, y)

# Make predictions
predictions = model.predict(np.array([[3.5, 3.5]]))
print(predictions)