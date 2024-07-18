import numpy as np
from keras.models import Sequential
from keras.layers import Dense

# Example data
X = np.array([[1, 1], [2, 2], [3, 3], [4, 4]])
y = np.array([0, 0, 1, 1])

# Create the model
model = Sequential()
model.add(Dense(12, input_dim=2, activation='relu'))
model.add(Dense(8, activation='relu'))
model.add(Dense(1, activation='sigmoid'))

# Compile the model
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

# Train the model
model.fit(X, y, epochs=150, batch_size=10, verbose=0)

# Make predictions
predictions = model.predict(np.array([[1.5, 1.5]]))
print(predictions)