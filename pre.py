import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from keras.models import load_model
import tensorflow as tf
import plotly.graph_objects as go
from flask import Flask, render_template, request
import yfinance as yf
from datetime import timedelta
import joblib

company_dict = dict(AMZN='Amazon', AAPL='Apple', GOOGL='Google', IBM='IBM', JPM='JPMorgan', NKE='NKE', TSLS='Tesla', AXP='American Express', BA='Boeing', CSCO='Cisco Systems', KO='Coca-Cola', GS='Goldman Sachs', INTC='Intel', WMT='Walmart', DIS='Walt Disney')

app = Flask(__name__)

def create_sliding_window(data, window_size):
    X, y = [], []
    for i in range(len(data) - window_size):
        X.append(data[i:i+window_size])
        y.append(data[i+window_size])
    return np.array(X), np.array(y)

def generate_prediction_graph(company):
    model = load_model("model10.h5")

    company_name = yf.Ticker(company)
    hist = company_name.history(period="1y")  # Retrieve 1 year of historical data
    filename = company + "_close_20.csv"
    stock_data = hist
    close_data_full = hist['Close']
    close_data = close_data_full.tail(290)  
    close_data.to_csv(filename)
    new_data = pd.read_csv(filename)

    scaler = MinMaxScaler(feature_range=(0, 1))
    new_scaled_prices = scaler.fit_transform(new_data[['Close']])

    sequence_length = 20

    def create_sequences(data, sequence_length):
        X = []
        y = []
        for i in range(len(data) - sequence_length):
            X.append(data[i:i+sequence_length])
            y.append(data[i+sequence_length])
        return np.array(X), np.array(y)

    X_new, y_new = create_sequences(new_scaled_prices, sequence_length)

    predictions = model.predict(X_new)

    predicted_prices = scaler.inverse_transform(predictions)

    original_dates = new_data['Date'].values[sequence_length:]
    original_prices = new_data['Close'].values[sequence_length:]

    last_date = pd.to_datetime(new_data['Date'].iloc[-1])
    next_dates = pd.date_range(start=last_date + timedelta(days=1), periods=20, freq='B')  # Use business days for next 20 days
    next_dates_str = [date.strftime('%Y-%m-%d') for date in next_dates]
    next_prices = []

    for i in range(15):
        window = new_scaled_prices[-sequence_length:]
        X = np.array([window])
        next_price = model.predict(X)
        next_prices.append(next_price[0, 0])
        new_scaled_prices = np.append(new_scaled_prices, next_price, axis=0)

    next_prices = scaler.inverse_transform([next_prices])[0]

    fig1 = go.Figure()
    fig1.add_trace(go.Scatter(x=original_dates, y=original_prices, name='Actual Data', mode='lines', line=dict(color='red')))
    fig1.add_trace(go.Scatter(x=next_dates_str, y=next_prices, name='Predicted Data', mode='lines', line=dict(color='blue')))
    fig1.update_layout(title=f'{company_dict.get(company)} Actual and Predicted Closing Prices using Gated Recurrent Unit',
                      xaxis_title='Date',
                      yaxis_title='Closing Price')

    model2 = joblib.load("linear_regression_model.joblib")

    stock_symbol = "AAPL"  # Example: Apple Inc.
    time_period = "1y"     # Fetching data for the last 1 year, adjust as needed

    # Retrieve historical data using yfinance
    stock_data = yf.download(stock_symbol, period=time_period)

    y = stock_data["Adj Close"].values
    dates = stock_data.index

    X = []
    for i in range(10, len(y)):
        X.append(y[i-10:i])

    X = np.array(X)  # Convert to NumPy array

    last_prices = X[-1]  # Last 10 days' prices from historical data

    # Generate predictions for the next 20 days
    next_20_days_predictions = []
    for _ in range(20):
        next_day_prediction = model2.predict([last_prices])[0]

        #next_day_prediction = model2.predict([last_prices.reshape(1, -1)])[0]
        next_20_days_predictions.append(next_day_prediction)
        last_prices = np.append(last_prices[1:], next_day_prediction)

    next_20_days_predictions = np.array(next_20_days_predictions)

    next_20 = []
    for _ in range(20):
        next_20.extend(next_20_days_predictions[_])
    # Generate dates for the predicted prices
    last_date = dates[-1]
    predicted_dates = [last_date + pd.DateOffset(days=i) for i in range(1, 21)]

    fig2 = go.Figure()
    fig2.add_trace(go.Scatter(x=dates, y=y, name="Actual Prices", mode="lines", line=dict(color='red')))
    fig2.add_trace(go.Scatter(x=predicted_dates, y=next_20, name="Predicted Prices", mode="lines", line=dict(color='blue')))
    fig2.update_layout(title=f'{company_dict.get(company)} Actual and Predicted Closing Prices using Linear Regression Model',
                    xaxis_title="Date",
                    yaxis_title="Price")

    return fig1.to_html(), fig2.to_html()


@app.route('/')
def home():
    return render_template('index.html')

@app.route('/graph')
def generate_graph():
    selected_company = request.args.get('company')
    prediction_graph1, prediction_graph2 = generate_prediction_graph(selected_company)
    return render_template('index.html', prediction_graph1=prediction_graph1, prediction_graph2=prediction_graph2)

if __name__ == '__main__':
    app.run()
