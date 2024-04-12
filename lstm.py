#lstm
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense

# Load your dataset into a pandas DataFrame
df = pd.read_csv('dataset.csv')

# Fill missing values with the previous data
df['BTC'].fillna(method='ffill', inplace=True)

# Extracting prices
prices = df['BTC'].values.reshape(-1, 1)

# Normalize the dataset
scaler = MinMaxScaler(feature_range=(0, 1))
prices_scaled = scaler.fit_transform(prices)

# Function to create the dataset with look back
def create_dataset(dataset, look_back=1):
    dataX, dataY = [], []
    for i in range(len(dataset) - look_back):
        a = dataset[i:(i + look_back), 0]
        dataX.append(a)
        dataY.append(dataset[i + look_back, 0])
    return np.array(dataX), np.array(dataY)

# Define the look back period (number of previous time steps to use for prediction)
look_back = 10

# Create the dataset with look back
X, y = create_dataset(prices_scaled, look_back)

# Reshape input to be [samples, time steps, features]
X = np.reshape(X, (X.shape[0], X.shape[1], 1))

# Define the LSTM model
model = Sequential()
model.add(LSTM(units=50, return_sequences=True, input_shape=(X.shape[1], 1)))
model.add(LSTM(units=50))
model.add(Dense(units=1))

# Compile the model
model.compile(optimizer='adam', loss='mean_squared_error')

# Fit the model
model.fit(X, y, epochs=100, batch_size=32)

# Predict the BTC price for 27th Jan
# Assuming you have the last 10 days' data available
last_10_days = prices[-look_back:]
scaled_last_10_days = scaler.transform(last_10_days.reshape(-1, 1))
input_data = scaled_last_10_days.reshape(1, look_back, 1)
predicted_price_scaled = model.predict(input_data)
predicted_price = scaler.inverse_transform(predicted_price_scaled)

print("Predicted BTC Price for next day for LSTM:", predicted_price[0][0])
