#App File
import streamlit as st
import pandas as pd
import altair as alt
import matplotlib.pyplot as plt
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
import plotly.express as px

st.set_page_config(
    page_title="PrediCrypt - Predict Crypto at ease",
    # page_icon="🏂",
    layout="wide",
    initial_sidebar_state="expanded")

st.markdown("""
<style>

[data-testid="block-container"] {
    padding-left: 2rem;
    padding-right: 2rem;
    padding-top: 1rem;
    padding-bottom: 0rem;
    margin-bottom: -7rem;
}

[data-testid="stVerticalBlock"] {
    padding-left: 0rem;
    padding-right: 0rem;
}

[data-testid="stMetric"] {
    background-color: #393939;
    text-align: center;
    padding: 15px 0;
}

[data-testid="stMetricLabel"] {
  display: flex;
  justify-content: center;
  align-items: center;
}

[data-testid="stMetricDeltaIcon-Up"] {
    position: relative;
    left: 38%;
    -webkit-transform: translateX(-50%);
    -ms-transform: translateX(-50%);
    transform: translateX(-50%);
}

[data-testid="stMetricDeltaIcon-Down"] {
    position: relative;
    left: 38%;
    -webkit-transform: translateX(-50%);
    -ms-transform: translateX(-50%);
    transform: translateX(-50%);
}
            
.reportview-container .sidebar-content {{
    padding-top: {1}rem;
}}
.reportview-container .main .block-container {{
    padding-top: {1}rem;
}}

</style>
""", unsafe_allow_html=True)

alt.themes.enable("dark")

dataframe = pd.read_csv('dataset.csv')
df_btc = pd.read_csv('dataset-BTC.csv')
df_ada = pd.read_csv('dataset-ADA.csv')
df_eth = pd.read_csv('dataset-ETH.csv')
df_ltc = pd.read_csv('dataset-LTC.csv')
df_usdt = pd.read_csv('dataset-USDT.csv')

def predict(dfname):
    df = pd.read_csv(dfname)

    # Fill missing values with the previous data
    df['Price'].fillna(method='ffill', inplace=True)

    # Extracting prices
    prices = df['Price'].values.reshape(-1, 1)

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
    model.fit(X, y, epochs=10, batch_size=32)

    # Predict the BTC price for 27th Jan
    # Assuming you have the last 10 days' data available
    last_10_days = prices[-look_back:]
    scaled_last_10_days = scaler.transform(last_10_days.reshape(-1, 1))
    input_data = scaled_last_10_days.reshape(1, look_back, 1)
    predicted_price_scaled = model.predict(input_data)
    predicted_price = scaler.inverse_transform(predicted_price_scaled)

    return predicted_price[0][0]

def make_graph(dfname,name):
    df = pd.read_csv(dfname)
    grp = df.groupby('Date',sort=False)
    avg = grp.mean('Price')
    fig =  plt.figure(figsize=(10,5), facecolor='#0e1117')
    # fig = plt.figure(facecolor='#0e1117')
    ax = plt.axes()
    plt.plot(avg.tail(15), color="#5c92ff")
    ax.set_facecolor("#0e1117")
    ax.set_xlabel('Date ',fontsize=16)
    ax.set_ylabel('Price in USD ',fontsize=16)
    ax.xaxis.label.set_color('white')        #setting up X-axis label color to yellow
    ax.yaxis.label.set_color('white') 
    ax.tick_params(colors='white')
    ax.spines['left'].set_color('white')
    ax.spines['bottom'].set_color('white')
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    plt.xticks(rotation=35)
    plt.title(name + " Price Trends",color='white',fontsize=16)
    return fig

with st.sidebar:
    
    st.image('logo1.png', caption='Predict Crypto at ease', width=160)
    st.title(' Cryptocurrencies')

    
    Coins = ['BTC', 'ETH', 'ADA', 'LTC', 'USDT']
    
    selected_coin = st.selectbox('Select a coin', Coins, index=len(Coins)-1)

    button = st.button("Predict")
    
    if button:
        result = predict('dataset-%s.csv'%selected_coin)
        st.markdown('# Predicted Value ')
        new_res = '<p style="font-family:sans-serif; color:#f7bd52; font-size: 42px;"> %.2f </p>' %(result)
        st.markdown(new_res,unsafe_allow_html=True)
    

    # color_theme_list = ['blues', 'cividis', 'greens', 'inferno', 'magma', 'plasma', 'reds', 'rainbow', 'turbo', 'viridis']
    # selected_color_theme = st.selectbox('Select a color theme', color_theme_list)
col = st.columns([1,0.5], gap='small')
with col[0]:
    st.markdown('#### Visualizing The Trends (Past 15 Days) ')

    chart = make_graph('dataset-%s.csv'%selected_coin, selected_coin)
    st.pyplot(chart)
    df_selected_coin = dataframe[selected_coin]
    

import seaborn as sn
with col[1]:
    df = pd.read_csv('dataset.csv')
    grp = df.groupby('Date',sort=False)
    data = grp.mean('Price').tail(16)
    data1 = data[['BTC','ETH','ADA','LTC','USDT']].pct_change()*100
    # print(data1)
    
    # plotting the heatmap 
    fig = px.imshow(data1[1:],aspect="auto")
    # hm = sn.heatmap(data = data1[1:]) 
    plt.tick_params(axis='both', colors='white')
    plt.title("Percentage Change",color='white',fontsize=16)
    # displaying the plotted heatmap 
    fig.update_layout(
    title="Percentage Change in Prices",
    xaxis_title="Cryptocurrencies",
    yaxis_title="Date",
    autosize=False,
    width=400,
    height=500,)
    st.plotly_chart(fig, theme=None)
#     df = pd.DataFrame({
#     'category': ['BTC', 'ETH', 'ADA', 'USDT', 'LTC'],
#     'value': [50, 15, 3, 100, 2]
#     })

#     # Create the pie chart
#     fig = px.pie(df, values='value', names='category', color_discrete_sequence=["yellow", "green", "blue", "red", "magenta"])

#     # Display the pie chart in Streamlit    
#     st.plotly_chart(fig)

# from numerize import numerize
def calc_diff(data):
    # diff = numerize.numerize(data.iloc[-1] - data.iloc[0])
    # res = "%.2f" % diff
    diff = data.iloc[-1] - data.iloc[0]
    # Format the difference with two decimal places
    formatted_diff = "{:.2f}".format(diff)
    return formatted_diff
    # return diff

from streamlit_extras.metric_cards import style_metric_cards
col1, col2, col3, col4, col5 = st.columns(5)

df1 = pd.read_csv('dataset.csv')
grp = df1.groupby('Date',sort=False)
data = grp.mean('Price').tail(15)
print(data.iloc[0])
print(data.iloc[-1])
col1.metric(label="Bitcoin", value="%.2f" % data['BTC'].iloc[-1], delta=calc_diff(data['BTC']))
col2.metric(label="Ether", value="%.2f" % data['ETH'].iloc[-1], delta=calc_diff(data['ETH']))
col3.metric(label="Cardano", value="%.2f" % data['ADA'].iloc[-1], delta=calc_diff(data['ADA']))
col4.metric(label="Litcoin", value="%.2f" % data['LTC'].iloc[-1], delta=calc_diff(data['LTC']))
col5.metric(label="Tether", value="%.2f" % data['USDT'].iloc[-1], delta=calc_diff(data['USDT']))


style_metric_cards(background_color='#0e1117')

