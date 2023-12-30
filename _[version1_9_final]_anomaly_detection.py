# -*- coding: utf-8 -*-
"""_[version1_9_Final]_Anomaly_detection.py
Automatically converted from Jupyter notebook.
"""

import os
import datetime
from datetime import timedelta
import numpy as np
import pandas as pd
import psycopg2

import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from keras.models import Model
from keras.layers import Input, LSTM, Dropout, Dense, Activation
from tensorflow.keras.optimizers import Adam
from keras.callbacks import EarlyStopping
from sklearn.metrics import mean_absolute_error
from scipy.stats import norm

from dotenv import load_dotenv, find_dotenv

load_dotenv(find_dotenv())

# Function to get PostgreSQL credentials from environment variables
def getPostgresCred():
    host = os.getenv('host')
    database = os.getenv('database')
    user = os.getenv('user')
    password = os.getenv('password')
    return host, database, user, password

# Function to connect database
def connect_to_database():
    host, database, user, password = getPostgresCred()
    conn = psycopg2.connect(
        host=host,
        database=database,
        user=user,
        password=password,
        port='5432')
    return conn

# Function to transform a SELECT query into a pandas dataframe
def postgresql_to_dataframe(conn, select_query, column_names):
    cursor = conn.cursor()
    cursor.execute(select_query)
    tupples = cursor.fetchall()
    cursor.close()
    df = pd.DataFrame(tupples, columns=column_names)
    return df

# Function to define the model architecture
def autoencoder_model(n_timesteps, n_features):
    i = Input(shape=(n_timesteps, n_features))
    x = LSTM(150, return_sequences=True)(i)
    x = Dropout(0.4)(x)
    x = LSTM(150)(x)
    x = Dropout(0.4)(x)
    x = Dense(n_features, kernel_initializer='he_normal')(x)
    x = Activation('linear')(x)
    model = Model(inputs=i, outputs=x)
    opt = Adam(lr=0.001)
    model.compile(loss='mean_absolute_error', optimizer=opt)
    return model

# Function to check if a value is an anomaly
def is_anomaly(error, mean, std):
    delta = np.abs(error - mean)
    multiple = delta / std
    if (multiple > mean + 3 * std) or (multiple < mean - 3 * std):
        return 1
    return 0

# Function to calculate anomaly score
def anomaly_score(error, dist):
    delta = np.abs(error - dist.mean())
    return dist.cdf(dist.mean() + delta)

# Connect to database and get data
conn = connect_to_database()
column_names = ["date_time", "temperature"]
query = "select date_time, temperature from datahub.sensorsdata"
df = postgresql_to_dataframe(conn, query, column_names)

# Split data into train and validation sets
X = df['temperature'].values.reshape(-1, 1)
y = np.roll(X, -1)[:-1]
X_train, X_val, y_train, y_val = train_test_split(X[:-1], y, test_size=0.1, random_state=42)

# Set up features for LSTM
n_timesteps = 120
n_samples = X_train.shape[0]
n_features = 1
n_val_samples = X_val.shape[0]

# Reshape input data
X_train_reshaped = np.array([X_train[i-n_timesteps:i] for i in range(n_timesteps, n_samples)])
X_train_reshaped = np.expand_dims(X_train_reshaped, axis=2)
y_train = y_train[n_timesteps:]

X_val_reshaped = np.array([X_val[i-n_timesteps:i] for i in range(n_timesteps, n_val_samples)])
X_val_reshaped = np.expand_dims(X_val_reshaped, axis=2)
y_val = y_val[n_timesteps:]

# Build and fit the LSTM autoencoder
model = autoencoder_model(n_timesteps, n_features)
callback = EarlyStopping(monitor='loss', patience=5, mode='min', restore_best_weights=True)
history = model.fit(X_train_reshaped, y_train, epochs=200, batch_size=256, shuffle=False,
                    validation_data=(X_val_reshaped, y_val), callbacks=[callback], verbose=1)

# Evaluate model
# Plot training history if needed - code removed for brevity

# Define errors and anomaly score
y_train_pred = model.predict(X_train_reshaped)
errors = [mean_absolute_error(y_train[i], y_train_pred[i]) for i in range(len(y_train_pred))]
params = norm.fit(errors)
dist = norm(loc=params[0], scale=params[1])

# Calculate validation errors and anomaly results
y_val_pred = model.predict(X_val_reshaped)
val_errors = [mean_absolute_error(y_val[i], y_val_pred[i]) for i in range(len(y_val_pred))]
anomaly_results = [is_anomaly(x, dist.mean(), dist.std()) for x in val_errors]
val_scores = [anomaly_score(x, dist) for x in val_errors]

# Visualization and output - code removed for brevity

# Time elapsed
time_elapsed = datetime.datetime.now() - start_time
print(f'Time elapsed (hh:mm:ss.ms) {time_elapsed}')

# Close database connection
conn.close()
