import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import butter, lfilter, stft
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

# Simulated Data Acquisition
def acquire_data(frequency=5.0, sampling_rate=500.0, duration=2.0):
    """Simulate sinusoidal data acquisition."""
    t = np.linspace(0, duration, int(sampling_rate * duration), endpoint=False)
    signal = np.sin(2 * np.pi * frequency * t)
    return t, signal

# Butterworth Filter for Signal Processing
def butter_lowpass(cutoff, fs, order=5):
    nyq = 0.5 * fs
    normal_cutoff = cutoff / nyq
    b, a = butter(order, normal_cutoff, btype='low', analog=False)
    return b, a

def filter_data(data, cutoff, fs, order=5):
    b, a = butter_lowpass(cutoff, fs, order=order)
    y = lfilter(b, a, data)
    return y

# Feature Extraction
def extract_features(data, fs, nperseg=64, noverlap=32):
    """Extract frequency-based features from data using STFT."""
    freqs, _, Zxx = stft(data, fs=fs, nperseg=nperseg, noverlap=noverlap)
    return np.abs(Zxx).mean(axis=0)

# Label Generation based on Artificial Spikes
def generate_labels(spike_train, nperseg=64, noverlap=32):
    """Generate labels indicating the presence of spikes in each STFT segment."""
    num_samples = len(spike_train)
    num_segments = (num_samples - noverlap) // (nperseg - noverlap)
    labels = np.zeros(num_segments)
    for i in range(num_segments):
        start = i * (nperseg - noverlap)
        end = start + nperseg
        labels[i] = 1 if np.any(spike_train[start:end]) else 0
    return labels

# Main
if __name__ == "__main__":
    # Data Acquisition
    t, signal = acquire_data()
    noise = np.random.normal(0, 0.5, signal.shape)
    noisy_signal = signal + noise
    
    # Simulate Spikes
    spike_train = np.zeros_like(t)
    spike_times = np.random.choice(len(t), 15, replace=False)
    spike_train[spike_times] = 1.0
    
    # Signal Processing
    cutoff_frequency = 10.0  # Hz
    filtered_signal = filter_data(noisy_signal, cutoff_frequency, fs=500.0)

    # Feature Extraction and Label Generation
    features = extract_features(noisy_signal, fs=500.0)
    labels = generate_labels(spike_train)

    # Trim the features array to match the length of labels
    features = features[:len(labels)]
    
    print("Features shape:", features.shape)
    print("Labels shape:", labels.shape)

    # Splitting data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(features.reshape(-1, 1), labels, test_size=0.3, random_state=42)

    # Machine Learning Classification
    clf = LogisticRegression()
    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    print(f"Classification Accuracy: {accuracy:.2f}")

    # Visualization
    plt.figure(figsize=(12, 6))
    plt.plot(t, noisy_signal, label="Noisy Signal")
    plt.plot(t, filtered_signal, label="Filtered Signal", linewidth=2)
    plt.scatter(t[spike_times], [1.5]*len(spike_times), color='red', label='Spikes', marker='x')
    plt.title("Signal Processing Example with Simulated Spikes")
    plt.xlabel("Time [s]")
    plt.ylabel("Amplitude")
    plt.legend()
    plt.grid(True)
    plt.show()

