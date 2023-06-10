import numpy as np
import pandas as pd
import yfinance as yf
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error
import matplotlib.pyplot as plt
import joblib


# Define the stock symbol and timeframe
stock_symbol = 'AAPL'  # Replace with the stock symbol you're interested in
start_date = '2010-01-01'
end_date = '2022-12-31'

# Retrieve stock price data from Yahoo Finance
data = yf.download(stock_symbol, start=start_date, end=end_date)
print(data)
# Preprocess the data
scaler = MinMaxScaler()
scaled_data = scaler.fit_transform(data['Close'].values.reshape(-1, 1))

# Split the data into training and testing sets
train_size = int(len(scaled_data) * 0.8)
train_data = scaled_data[:train_size]
test_data = scaled_data[train_size:]

# Create sequences for training
sequence_length = 10

def create_sequences(data, sequence_length):
    X = []
    y = []
    for i in range(len(data) - sequence_length):
        X.append(data[i:i+sequence_length])
        y.append(data[i+sequence_length])
    return np.array(X), np.array(y)

X_train, y_train = create_sequences(train_data, sequence_length)
X_test, y_test = create_sequences(test_data, sequence_length)

# Flatten the sequences
X_train = X_train.reshape(-1, sequence_length)
X_test = X_test.reshape(-1, sequence_length)

# Build the linear regression model
model = LinearRegression()

# Train the model
model.fit(X_train, y_train)

model_filename = 'linear_regression_model.joblib'
joblib.dump(model, model_filename)

# Make predictions
predicted_data = model.predict(X_test)
predicted_prices = scaler.inverse_transform(predicted_data.reshape(-1, 1))

predicted_data = model.predict(X_test)
predicted_prices = scaler.inverse_transform(predicted_data.reshape(-1, 1))

# Determine the direction of price movement
actual_direction = np.sign(y_test - X_test[:, -1])
predicted_direction = np.sign(predicted_data - X_test[:, -1])

# Calculate the accuracy percentage
accuracy = (predicted_direction == actual_direction).mean() * 100
print(f"Accuracy: {accuracy:.2f}%")

# Evaluate the model
mse = mean_squared_error(y_test, predicted_data)
print(f"Mean Squared Error: {mse}")

# Visualize the results
plt.plot(data['Close'].values[train_size+sequence_length:], label='Actual')
plt.plot(predicted_prices, label='Predicted')
plt.legend()
plt.show()
