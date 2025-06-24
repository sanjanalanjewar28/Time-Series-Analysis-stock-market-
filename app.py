import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import datetime
from data_loader import load_stock_data
from models import create_prophet_model, create_lstm_model
from visualization import (
    plot_stock_data, 
    plot_prophet_forecast, 
    plot_lstm_results, 
    create_interactive_plot
)

def main():
    # Set page title
    st.set_page_config(
        page_title="Stock Market Analysis & Forecasting",
        page_icon="📈",
        layout="wide"
    )
    
    # Add header
    st.title("📈 Stock Market Analysis & Forecasting")
    st.write("""
    This application allows you to analyze and forecast stock market trends using time series analysis techniques.
    Select a stock symbol, date range, and forecasting model to get started.
    """)
    
    # Sidebar for inputs
    st.sidebar.header("Input Parameters")
    
    # Stock ticker input
    ticker = st.sidebar.text_input("Stock Symbol (e.g., AAPL, MSFT, GOOGL)", "AAPL")
    
    # Date range input
    today = datetime.datetime.now().date()
    five_years_ago = today - datetime.timedelta(days=5*365)
    
    start_date = st.sidebar.date_input("Start Date", five_years_ago)
    end_date = st.sidebar.date_input("End Date", today)
    
    # Forecasting parameters
    st.sidebar.header("Forecasting Parameters")
    forecast_days = st.sidebar.slider("Forecast Days", 7, 365, 30)
    
    # Model selection
    model_option = st.sidebar.selectbox(
        "Forecasting Model",
        ["Prophet", "LSTM", "Both"]
    )
    
    # Load data button
    if st.sidebar.button("Load Data and Forecast"):
        # Load stock data
        with st.spinner("Loading stock data..."):
            data = load_stock_data(ticker, start_date, end_date)
        
        if data is not None and not data.empty:
            # Display success message
            st.success(f"Successfully loaded data for {ticker}!")
            
            # Display data overview
            st.header("Data Overview")
            st.dataframe(data.head())
            
            # Display basic statistics
            st.subheader("Basic Statistics")
            st.dataframe(data.describe())
            
            # Plot stock data
            st.header("Stock Price History")
            fig = plot_stock_data(data, title=f"{ticker} Stock Price History")
            st.pyplot(fig)
            
            # Forecasting section
            st.header("Forecasting Results")
            
            # Run selected models
            prophet_results = None
            lstm_results = None
            
            if model_option in ["Prophet", "Both"]:
                with st.spinner("Running Prophet model..."):
                    st.subheader("Prophet Model Forecast")
                    prophet_model, prophet_forecast = create_prophet_model(data, forecast_period=forecast_days)
                    fig_components, fig_forecast = plot_prophet_forecast(prophet_model, prophet_forecast, data)
                    
                    st.pyplot(fig_forecast)
                    st.pyplot(fig_components)
                    prophet_results = prophet_forecast
            
            if model_option in ["LSTM", "Both"]:
                with st.spinner("Running LSTM model..."):
                    st.subheader("LSTM Model Forecast")
                    sequence_length = 60  # Number of days to look back
                    lstm_model, predictions, future_predictions, scaler = create_lstm_model(
                        data, 
                        target_column='Close', 
                        sequence_length=sequence_length,
                        forecast_days=forecast_days
                    )
                    
                    fig_lstm = plot_lstm_results(
                        data, 
                        predictions, 
                        future_predictions, 
                        sequence_length
                    )
                    st.pyplot(fig_lstm)
                    
                    # Create future dates for LSTM predictions
                    last_date = data.index[-1]
                    future_dates = pd.date_range(
                        start=last_date + pd.Timedelta(days=1), 
                        periods=len(future_predictions)
                    )
                    
                    lstm_results = (predictions, future_predictions, future_dates)
            
            # Interactive comparison plot
            if model_option == "Both":
                st.header("Model Comparison")
                fig_interactive = create_interactive_plot(data, prophet_forecast, lstm_results)
                st.plotly_chart(fig_interactive, use_container_width=True)
            
            # Stock analysis insights
            st.header("Stock Analysis Insights")
            
            # Calculate moving averages
            data['MA50'] = data['Close'].rolling(window=50).mean()
            data['MA200'] = data['Close'].rolling(window=200).mean()
            
            # Plot moving averages
            fig, ax = plt.subplots(figsize=(12, 6))
            ax.plot(data.index, data['Close'], label='Close Price')
            ax.plot(data.index, data['MA50'], label='50-day MA')
            ax.plot(data.index, data['MA200'], label='200-day MA')
            ax.set_title(f"{ticker} Stock Price with Moving Averages")
            ax.set_xlabel('Date')
            ax.set_ylabel('Price')
            ax.grid(True)
            ax.legend()
            st.pyplot(fig)
            
            # Calculate volatility
            data['Daily Return'] = data['Close'].pct_change()
            volatility = data['Daily Return'].std() * (252 ** 0.5)  # Annualized volatility
            
            # Display volatility
            st.subheader("Volatility Analysis")
            st.write(f"Annualized Volatility: {volatility:.2%}")
            
            # Plot daily returns
            fig, ax = plt.subplots(figsize=(12, 6))
            ax.plot(data.index, data['Daily Return'], label='Daily Returns')
            ax.set_title(f"{ticker} Daily Returns")
            ax.set_xlabel('Date')
            ax.set_ylabel('Return')
            ax.grid(True)
            ax.legend()
            st.pyplot(fig)
            
            # Display correlation matrix for OHLC data
            st.subheader("Correlation Matrix")
            corr = data[['Open', 'High', 'Low', 'Close', 'Volume']].corr()
            fig, ax = plt.subplots(figsize=(10, 8))
            sns.heatmap(corr, annot=True, cmap='coolwarm', ax=ax)
            st.pyplot(fig)
            
        else:
            st.error(f"Failed to load data for {ticker}. Please check the ticker symbol and try again.")

if __name__ == "__main__":
    main()