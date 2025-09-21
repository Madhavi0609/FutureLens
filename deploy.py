#!/usr/bin/env python
# coding: utf-8

import streamlit as st
import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
import matplotlib.pyplot as plt

st.title("Forecasting of Hindustan Petroleum Oil Price Using LSTM :")

# Load your dataset
Apple = pd.read_csv('HINDPETRO.NS.csv')
columns_to_drop1 = ["Open", "High", "Low", "Adj Close", "Volume"]
df = Apple.drop(columns=columns_to_drop1)
df.rename(columns={'Date': 'ds', 'Close': 'y'}, inplace=True)

# Handle missing values
df.dropna(inplace=True)

# Normalize the data
scaler = MinMaxScaler()
df['y'] = scaler.fit_transform(df[['y']])

# Function to prepare data for LSTM
def prepare_data(data, time_steps):
    X, y = [], []
    for i in range(len(data) - time_steps):
        X.append(data[i:(i + time_steps), 0])
        y.append(data[i + time_steps, 0])
    return np.array(X), np.array(y)


time_steps = 10  # Coinsdering the time_steps
X, y = prepare_data(df[['y']].values, time_steps)

# Exclude Sundays and Saturdays from the data
df['weekday'] = pd.to_datetime(df['ds']).dt.weekday
df = df[(df['weekday'] != 5) & (df['weekday'] != 6)].copy()

# Reshape input to be [samples, time steps, features]
X = np.reshape(X, (X.shape[0], X.shape[1], 1))

import tensorflow as tf

physical_devices = tf.config.list_physical_devices('GPU')
if physical_devices:
    tf.config.experimental.set_memory_growth(physical_devices[0], True)

# Build the LSTM model
model = Sequential()
model.add(LSTM(units=25, return_sequences=True, input_shape=(X.shape[1], 1)))
model.add(LSTM(units=25, return_sequences=False))
model.add(Dense(units=1))
model.compile(optimizer='adam', loss='mean_squared_error')

# Train the model
model.fit(X, y, epochs=10, batch_size=128)

# Date input from the user
date_input = st.date_input("Enter a date for forecasting (excluding Sundays and Saturdays):", pd.to_datetime('today'))

forecast_date = pd.to_datetime(date_input).replace(tzinfo=None)

# Exclude Sundays and Saturdays from the forecasting data
if date_input.weekday() == 5 or date_input.weekday() == 6:
    st.warning("Please choose a weekday (excluding Sundays and Saturdays) for forecasting.")
else:
    # Prepare data for forecasting
    last_data = df['y'].values[-time_steps:]
    last_data_scaled = scaler.transform(last_data.reshape(-1, 1))
    last_data_reshaped = np.reshape(last_data, (1, time_steps, 1))

    # Forecast
    forecast = model.predict(last_data_reshaped)
    forecast = scaler.inverse_transform(forecast)[0, 0]

    # Display the forecast
    st.write('Forecast:')
    st.write(forecast)
