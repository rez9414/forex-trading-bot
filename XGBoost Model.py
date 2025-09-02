import pandas as pd
import numpy as np
from xgboost import XGBRegressor
from sklearn.model_selection import train_test_split

# Load data
data = pd.read_csv('historical_eurusd.csv')
data['Volatility'] = data['Close'].pct_change().rolling(20).std()

# Features: lagged prices, technical indicators
data['Lag1'] = data['Close'].shift(1)
data['Lag2'] = data['Close'].shift(2)
data = data.dropna()

X = data[['Lag1', 'Lag2']]
y = data['Volatility']

# Split data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=False)

# Build XGBoost model
model = XGBRegressor(n_estimators=100, learning_rate=0.1, max_depth=5)
model.fit(X_train, y_train)

# Predict
predictions = model.predict(X_test)

print("Sample Volatility Prediction:", predictions[0])
