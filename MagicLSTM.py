import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split

# Set a seed for reproducibility
np.random.seed(42)
tf.random.set_seed(42)

# Sample data creation (you will replace this with your actual game state data)
# Let's assume each card has some features: card_id, card_type, triggered_ability, passive_ability
# For demonstration, let’s generate synthetic data
num_samples = 1000  # number of hands
num_cards = 10  # cards per hand

# Generate sample data (in actuality, you'll load your MTG card data)
data = []
for _ in range(num_samples):
    hand = np.random.randint(1, 3000, num_cards)  # fake card IDs
    triggered = np.random.randint(0, 2, num_cards)  # 0 or 1 for triggered ability
    passive = np.random.randint(0, 2, num_cards)  # 0 or 1 for passive ability
    effective_value = np.random.random()  # a measure of hand effectiveness
    data.append(np.concatenate([hand, triggered, passive, [effective_value]]))

# Creating a DataFrame
df = pd.DataFrame(data, columns=[f'card_{i}' for i in range(num_cards)] + 
                                  [f'triggered_{i}' for i in range(num_cards)] + 
                                  [f'passive_{i}' for i in range(num_cards)] + 
                                  ['effective_value']])

# Features and labels
X = df.drop('effective_value', axis=1).values
y = df['effective_value'].values

# Reshape input for LSTM network [samples, time_steps, features]
X = X.reshape((X.shape[0], 1, X.shape[1]))  # Here, time_steps = 1

# Split data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

# Build LSTM model
model = Sequential()
model.add(LSTM(50, activation='relu', input_shape=(X.shape[1], X.shape[2])))
model.add(Dropout(0.2))
model.add(Dense(1))  # output layer for regression

model.compile(optimizer='adam', loss='mean_squared_error')

# Fit model
model.fit(X_train, y_train, epochs=100, batch_size=32, validation_data=(X_test, y_test))

# Making predictions
predictions = model.predict(X_test)

# To assess performance, you might use:
from sklearn.metrics import mean_squared_error
mse = mean_squared_error(y_test, predictions)
print(f'Mean Squared Error: {mse}')
