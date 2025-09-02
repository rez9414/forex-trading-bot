import numpy as np
import pandas as pd
from tensorflow.keras.layers import Input, Dense, MultiHeadAttention, LayerNormalization
from tensorflow.keras.models import Model
from sklearn.preprocessing import MinMaxScaler

# Load data
data = pd.read_csv('historical_eurusd.csv')
prices = data['Close'].values.reshape(-1, 1)
sentiment = data['Sentiment'].values.reshape(-1, 1)  # Hypothetical sentiment score

# Normalize
scaler = MinMaxScaler()
inputs = np.hstack([prices, sentiment])
inputs = scaler.fit_transform(inputs)

# Prepare sequences
def create_sequences(data, seq_length):
    X, y = [], []
    for i in range(len(data) - seq_length):
        X.append(data[i:i + seq_length])
        y.append(data[i + seq_length, 0])  # Predict price
    return np.array(X), np.array(y)

seq_length = 60
X, y = create_sequences(inputs, seq_length)

# Build Transformer
inputs = Input(shape=(seq_length, 2))
x = MultiHeadAttention(num_heads=4, key_dim=2)(inputs, inputs)
x = LayerNormalization(epsilon=1e-6)(x + inputs)
x = Dense(64, activation='relu')(x)
x = Dense(1)(x)
model = Model(inputs, x)

model.compile(optimizer='adam', loss='mean_squared_error')
model.fit(X, y, epochs=10, batch_size=32)

# Predict
predictions = model.predict(X)
print("Sample Transformer Prediction:", predictions[0])
