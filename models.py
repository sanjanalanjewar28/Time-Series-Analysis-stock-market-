import pandas as pd
import numpy as np
from prophet import Prophet
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
import matplotlib.pyplot as plt

def create_prophet_model(data, forecast_period=30):
    """
    Create and train a Prophet model for time series forecasting
    
    Parameters:
    -----------
    data : pandas.DataFrame
        DataFrame containing stock data with DatetimeIndex
    forecast_period : int
        Number of days to forecast
        
    Returns:
    --------
    tuple
        (Prophet model, forecast DataFrame)
    """
    # Prepare data for Prophet
    df_prophet = data.reset_index()
    df_prophet = df_prophet.rename(columns={'Date': 'ds', 'Close': 'y'})
    
    # Create and train Prophet model
    model = Prophet()
    model.fit(df_prophet)
    
    # Create future dataframe
    future = model.make_future_dataframe(periods=forecast_period)
    
    # Make predictions
    forecast = model.predict(future)
    
    return model, forecast

def create_lstm_model(data, target_column='Close', sequence_length=60, forecast_days=30, epochs=50, batch_size=32):
    """
    Create and train an LSTM model for time series forecasting
    
    Parameters:
    -----------
    data : pandas.DataFrame
        DataFrame containing stock data
    target_column : str
        Column to predict (default: 'Close')
    sequence_length : int
        Number of time steps to look back
    forecast_days : int
        Number of days to forecast
    epochs : int
        Number of training epochs
    batch_size : int
        Batch size for training
        
    Returns:
    --------
    tuple
        (LSTM model, predictions, future_predictions, scaler)
    """
    # Extract the target column and convert to numpy array
    dataset = data[target_column].values.reshape(-1, 1)
    
    # Scale the data
    scaler = MinMaxScaler(feature_range=(0, 1))
    scaled_data = scaler.fit_transform(dataset)
    
    # Create sequences
    X, y = [], []
    for i in range(sequence_length, len(scaled_data)):
        X.append(scaled_data[i-sequence_length:i, 0])
        y.append(scaled_data[i, 0])
    
    X, y = np.array(X), np.array(y)
    X = np.reshape(X, (X.shape[0], X.shape[1], 1))
    
    # Split data into train and test sets
    train_size = int(len(X) * 0.8)
    X_train, X_test = X[:train_size], X[train_size:]
    y_train, y_test = y[:train_size], y[train_size:]
    
    # Create LSTM model
    model = Sequential()
    model.add(LSTM(units=50, return_sequences=True, input_shape=(X_train.shape[1], 1)))
    model.add(Dropout(0.2))
    model.add(LSTM(units=50, return_sequences=False))
    model.add(Dropout(0.2))
    model.add(Dense(units=1))
    
    # Compile and train the model
    model.compile(optimizer='adam', loss='mean_squared_error')
    model.fit(X_train, y_train, epochs=epochs, batch_size=batch_size, verbose=1)
    
    # Make predictions
    predictions = model.predict(X_test)
    predictions = scaler.inverse_transform(predictions)
    
    # Forecast future values
    last_sequence = scaled_data[-sequence_length:].reshape(1, sequence_length, 1)
    future_predictions = []
    
    for _ in range(forecast_days):
        # Predict the next value
        next_pred = model.predict(last_sequence)[0][0]
        future_predictions.append(next_pred)
        
        # Update the sequence
        last_sequence = np.append(last_sequence[:, 1:, :], [[next_pred]], axis=1)
    
    # Inverse transform the predictions
    future_predictions = np.array(future_predictions).reshape(-1, 1)
    future_predictions = scaler.inverse_transform(future_predictions)
    
    return model, predictions, future_predictions, scaler