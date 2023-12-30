# """
#     Here is clean version of code. But I did not use all the techniques used in jupyterlab.
# """

import os
import datetime
import numpy as np
import pandas as pd
import psycopg2
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error
from scipy.stats import norm
from keras.layers import Input, LSTM, Dropout, Dense, Activation
from keras.models import Model
from tensorflow.keras.optimizers import Adam
from keras.callbacks import EarlyStopping
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Database credentials should not be hardcoded. Instead, use environment variables or a config file.
def getPostgresCred():
    host = os.getenv('DB_HOST')
    database = os.getenv('DB_DATABASE')
    user = os.getenv('DB_USER')
    password = os.getenv('DB_PASSWORD')
    port = os.getenv('PORT')
    return host, database, user, password, port

def connect_to_database():
    host, database, user, password = getPostgresCred()
    return psycopg2.connect(host=host, database=database, user=user, password=password, port=port)

def fetch_data(conn, select_query, column_names):
    with conn.cursor() as cursor:
        cursor.execute(select_query)
        tupples = cursor.fetchall()
        return pd.DataFrame(tupples, columns=column_names)


# Define autoencoder architecture
def build_autoencoder(n_timesteps, n_features, lstm_units, dropout):
    input_layer = Input(shape=(n_timesteps, n_features))
    encoder = LSTM(lstm_units, return_sequences=True)(input_layer)
    encoder = Dropout(dropout)(encoder)
    encoder = LSTM(lstm_units)(encoder)
    encoder = Dropout(dropout)(encoder)
    decoder = Dense(n_features, kernel_initializer='he_normal')(encoder)
    output_layer = Activation('linear')(decoder)

    model = Model(inputs=input_layer, outputs=output_layer)
    model.compile(loss='mean_absolute_error', optimizer=Adam(lr=0.001))
    return model


def reshape_series(series, n_timesteps):
    return np.array([series[i - n_timesteps:i] for i in range(n_timesteps, len(series))])


def train_model(model, X_train, y_train, X_val, y_val):
    callback = EarlyStopping(monitor='loss', patience=5, restore_best_weights=True)
    history = model.fit(X_train, y_train, epochs=200, batch_size=256, callbacks=[callback], validation_data=(X_val, y_val), verbose=1)
    return history


# Start of main script
start_time = datetime.datetime.now()

# Connect to the database
conn = connect_to_database()

# Fetch data from the database
query = "SELECT date_time, temperature FROM datahub.sensorsdata"
df = fetch_data(conn, query, ['date_time', 'temperature'])

# Data Preprocessing
data = df[:15500]  # Sample data
X = data['temperature'].values.reshape(-1, 1)
y = np.roll(X, -1)[:-1][:-1]  # Align y to the next value in X
X_train, X_val, y_train, y_val = train_test_split(X[:-1], y, test_size=0.1, random_state=42)

# Prepare data for LSTM Autoencoder
n_timesteps = 120
X_train_reshaped = reshape_series(X_train, n_timesteps)
X_train_reshaped = np.expand_dims(X_train_reshaped, axis=2)
X_val_reshaped = reshape_series(X_val, n_timesteps)
X_val_reshaped = np.expand_dims(X_val_reshaped, axis=2)

# Build and train the autoencoder
autoencoder = build_autoencoder(n_timesteps, 1, 150, 0.4)
history = train_model(autoencoder, X_train_reshaped, y_train, X_val_reshaped, y_val)

# Model Evaluation
plt.plot(history.history['loss'], label='Train Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.title('Model Loss During Training')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()
plt.show()

# Detect anomalies
y_train_pred = autoencoder.predict(X_train_reshaped)
train_errors = mean_absolute_error(y_train, y_train_pred, multioutput='raw_values')
dist = norm(*norm.fit(train_errors))

# Function to decide if a data point is anomalous
def is_anomaly(error, dist_threshold):
    return int(error > dist_threshold)

# Define a threshold for anomaly detection
threshold = dist.mean() + 3 * dist.std()

# Map errors through the anomaly detection function
anomalies = np.array([is_anomaly(error, threshold) for error in train_errors])

# Visualize the anomalies
sns.displot(anomalies, kde=True, height=5, aspect=2)
plt.title('Anomaly Scores')
plt.show()

# Time elapsed
time_elapsed = datetime.datetime.now() - start_time
print('Time elapsed (hh:mm:ss.ms) {}'.format(time_elapsed))

# Close database connection
conn.close()