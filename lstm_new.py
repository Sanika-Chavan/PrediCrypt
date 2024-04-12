import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense


data = pd.read_csv('dataset.csv')

today = pd.to_datetime('today').date().strftime('%d-%m-%Y')  # Get today's date
print(today)
# Get the index of the row with the previous date
# previous_date_index = data[data['Date'] == today - pd.Timedelta(days=1)].index[0]

# # Get data up to the previous date (excluding today)
# df = data.iloc[:previous_date_index + 1] 
df = data.query("Date < @today")

df['BTC'].fillna(method='ffill', inplace=True)
prices_LSTM_BTC = df['BTC'].values.reshape(-1, 1)

# Normalize the dataset
scaler = MinMaxScaler(feature_range=(0, 1))
prices_scaled = scaler.fit_transform(prices_LSTM_BTC)

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
print(X.size)

# Define the LSTM model
model = Sequential()
model.add(LSTM(units=50, return_sequences=True, input_shape=(X.shape[1], 1)))
model.add(LSTM(units=50))
model.add(Dense(units=1))

# Compile the model
model.compile(optimizer='adam', loss='mean_squared_error')

# Fit the model
model.fit(X, y, epochs=20, batch_size=32)

# Assuming you have the last 10 days' data available
last_10_days = prices_LSTM_BTC[-look_back:]
scaled_last_10_days = scaler.transform(last_10_days.reshape(-1, 1))
input_data = scaled_last_10_days.reshape(1, look_back, 1)
predicted_price_scaled = model.predict(input_data)
predicted_price_btc_prev = scaler.inverse_transform(predicted_price_scaled)

print("Predicted BTC Price for today for LSTM BTC:", predicted_price_btc_prev)

grp = data['Date','BTC'].groupby('Date',sort=False)
# print(grp)
# td_df = grp[grp['Date'] == today]
avg = grp.mean('Price')
# print(avg)
actual = avg[today]

print("Prediction: ", predicted_price_btc_prev)
print("Actual: ",actual)
print("Difference: ",predicted_price_btc_prev-actual)
mae = np.mean(np.abs(predicted_price_btc_prev - actual))
print("Mean Absolute Error (MAE):", mae)