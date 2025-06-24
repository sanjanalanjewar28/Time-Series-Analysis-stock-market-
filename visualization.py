import matplotlib.pyplot as plt
import plotly.graph_objs as go
import pandas as pd
import numpy as np

def plot_stock_data(data, title='Stock Price History'):
    """
    Plot stock price history using Matplotlib.
    """
    if data.empty:
        raise ValueError("DataFrame is empty. Cannot plot.")
    if 'Close' not in data.columns:
        raise ValueError("'Close' column missing from data.")
    if not isinstance(data.index, pd.DatetimeIndex):
        raise TypeError("Data index must be a DatetimeIndex.")

    fig, ax = plt.subplots(figsize=(12, 6))
    ax.plot(data.index, data['Close'], label='Close Price')
    ax.set_title(title)
    ax.set_xlabel('Date')
    ax.set_ylabel('Price')
    ax.grid(True)
    ax.legend()
    
    return fig

def plot_prophet_forecast(model, forecast, data):
    """
    Plot Prophet forecast including components and actual data points.
    """
    required_cols = ['ds', 'yhat', 'yhat_upper', 'yhat_lower']
    for col in required_cols:
        if col not in forecast.columns:
            raise ValueError(f"Column '{col}' missing in Prophet forecast data")

    if 'Close' not in data.columns:
        raise ValueError("'Close' column missing from original data.")
    if not isinstance(data.index, pd.DatetimeIndex):
        raise TypeError("Data index must be a DatetimeIndex.")

    fig_components = model.plot_components(forecast)
    fig_forecast = model.plot(forecast)
    
    ax = fig_forecast.gca()
    ax.plot(data.index, data['Close'], 'ko', alpha=0.3, label='Actual')
    ax.legend()
    
    return fig_components, fig_forecast

def plot_lstm_results(data, predictions, future_predictions, sequence_length, target_column='Close'):
    """
    Plot LSTM model predictions alongside actual data.
    """
    if data.empty:
        raise ValueError("Data is empty.")
    if target_column not in data.columns:
        raise ValueError(f"'{target_column}' column missing in data.")
    if not isinstance(data.index, pd.DatetimeIndex):
        raise TypeError("Data index must be a DatetimeIndex.")

    if not isinstance(predictions, (np.ndarray, pd.Series)) or predictions.ndim not in [1, 2]:
        raise TypeError("LSTM 'predictions' must be a 1D or 2D array.")
    if not isinstance(future_predictions, (np.ndarray, pd.Series)) or future_predictions.ndim not in [1, 2]:
        raise TypeError("LSTM 'future_predictions' must be a 1D or 2D array.")

    last_date = data.index[-1]
    future_dates = pd.date_range(start=last_date + pd.Timedelta(days=1), periods=len(future_predictions))

    fig, ax = plt.subplots(figsize=(12, 6))
    train_size = int(len(data) * 0.8)

    ax.plot(data.index[:train_size], data[target_column][:train_size], label='Training Data')
    ax.plot(data.index[train_size:], data[target_column][train_size:], label='Test Data')

    test_dates = data.index[train_size + sequence_length:]
    ax.plot(test_dates, predictions.flatten(), label='LSTM Predictions', alpha=0.7)
    ax.plot(future_dates, future_predictions.flatten(), label='Future Predictions', color='red')

    ax.set_title('LSTM Model: Actual vs Predicted')
    ax.set_xlabel('Date')
    ax.set_ylabel('Price')
    ax.grid(True)
    ax.legend()

    return fig

def create_interactive_plot(data, prophet_forecast=None, lstm_predictions=None):
    """
    Create an interactive Plotly visualization of stock data, forecasts, and predictions.
    """
    if not isinstance(data, pd.DataFrame):
        raise TypeError("data must be a pandas DataFrame")

    required_cols = ['Close', 'Open', 'High', 'Low']
    missing_cols = [col for col in required_cols if col not in data.columns]
    if missing_cols:
        raise ValueError(f"Missing columns in data: {missing_cols}")

    if not isinstance(data.index, pd.DatetimeIndex):
        raise TypeError("Data index must be a DatetimeIndex")

    fig = go.Figure()

    fig.add_trace(go.Scatter(
        x=data.index,
        y=data['Close'],
        mode='lines',
        name='Actual Close Price',
        line=dict(color='blue')
    ))

    fig.add_trace(go.Candlestick(
        x=data.index,
        open=data['Open'],
        high=data['High'],
        low=data['Low'],
        close=data['Close'],
        name='OHLC',
        visible='legendonly'
    ))

    if prophet_forecast is not None:
        for col in ['ds', 'yhat', 'yhat_upper', 'yhat_lower']:
            if col not in prophet_forecast:
                raise ValueError(f"'{col}' missing from Prophet forecast")

        fig.add_trace(go.Scatter(
            x=prophet_forecast['ds'],
            y=prophet_forecast['yhat'],
            mode='lines',
            name='Prophet Forecast',
            line=dict(color='green')
        ))

        fig.add_trace(go.Scatter(
            x=prophet_forecast['ds'],
            y=prophet_forecast['yhat_upper'],
            mode='lines',
            name='Prophet Upper Bound',
            line=dict(width=0),
            showlegend=False
        ))

        fig.add_trace(go.Scatter(
            x=prophet_forecast['ds'],
            y=prophet_forecast['yhat_lower'],
            mode='lines',
            name='Prophet Lower Bound',
            line=dict(width=0),
            fill='tonexty',
            fillcolor='rgba(0, 176, 0, 0.2)',
            showlegend=False
        ))

    if lstm_predictions is not None:
        predictions, future_predictions, future_dates = lstm_predictions
        train_size = int(len(data) * 0.8)
        sequence_length = 60

        test_dates = data.index[train_size + sequence_length:]
        fig.add_trace(go.Scatter(
            x=test_dates,
            y=predictions.flatten(),
            mode='lines',
            name='LSTM Test Predictions',
            line=dict(color='orange')
        ))

        fig.add_trace(go.Scatter(
            x=future_dates,
            y=future_predictions.flatten(),
            mode='lines',
            name='LSTM Future Forecast',
            line=dict(color='red')
        ))

    fig.update_layout(
        title='Stock Price Analysis and Forecast',
        xaxis_title='Date',
        yaxis_title='Price',
        hovermode='x unified',
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        template='plotly_white'
    )

    return fig
