import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import butter, lfilter
import plotly.graph_objects as go
import tensorflow as tf
from sklearn.model_selection import train_test_split

# Simulated Data Acquisition
def acquire_data(frequency=5.0, sampling_rate=500.0, duration=2.0):
    t = np.linspace(0, duration, int(sampling_rate * duration), endpoint=False)
    signal = np.sin(2 * np.pi * frequency * t)
    return t, signal

def butter_bandpass(lowcut, highcut, fs, order=5):
    nyq = 0.5 * fs
    low = lowcut / nyq
    high = highcut / nyq
    b, a = butter(order, [low, high], btype='band')
    return b, a

def bandpass_filter(data, lowcut, highcut, fs, order=5):
    b, a = butter_bandpass(lowcut, highcut, fs, order=order)
    y = lfilter(b, a, data)
    return y

# Neural Network for Spike Detection
def simple_nn_model(input_shape):
    model = tf.keras.Sequential([
        tf.keras.layers.Dense(32, activation='relu', input_shape=(input_shape,)),
        tf.keras.layers.Dense(16, activation='relu'),
        tf.keras.layers.Dense(1, activation='sigmoid')
    ])
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    return model

if __name__ == "__main__":
    t, signal = acquire_data()
    noise = np.random.normal(0, 0.5, signal.shape)
    noisy_signal = signal + noise

    # Bandpass filter the signal
    bandpass_signal = bandpass_filter(noisy_signal, 1.0, 10.0, 500.0)

    # Generate a simple neural network model
    features = bandpass_signal
    labels = (signal > 0).astype(int)  # Spike detection (not real, just for demonstration)
    model = simple_nn_model(1)
    X_train, X_test, y_train, y_test = train_test_split(features.reshape(-1, 1), labels, test_size=0.3)
    model.fit(X_train, y_train, epochs=10, batch_size=32, validation_data=(X_test, y_test))

    # Interactive visualization using Plotly
    fig = go.Figure()
    fig.add_trace(go.Scatter(y=noisy_signal, mode='lines', name='Noisy Signal'))
    fig.add_trace(go.Scatter(y=bandpass_signal, mode='lines', name='Bandpass Filtered Signal'))
    fig.show()
