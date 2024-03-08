import pandas as pd
import matplotlib.pyplot as plt
from scipy import signal
import numpy as np
import seaborn as sns
from scipy.stats import mannwhitneyu
from scipy.signal import butter, filtfilt

# Read the CSV file into a DataFrame
file_path_pickup = 'pickup_data1_20240124_110706.csv'
file_path_walk = 'walk_data1_20240124_103004.csv'

# Read the files
pickup_data = pd.read_csv(file_path_pickup)
walk_data = pd.read_csv(file_path_walk)

def plot_sensor_data(df, activity):
    # Convert timestamp to datetime
    df['timestamp'] = pd.to_datetime(df['timestamp'])

    # Create a figure and a set of subplots
    fig, ax1 = plt.subplots(figsize=(12, 6))

    # Plot acceleration data
    ax1.set_xlabel('Time')
    ax1.set_ylabel('Acceleration', color='tab:blue')
    ax1.plot(df['timestamp'], df['acceleration_x'], label='Acceleration X', color='tab:blue', alpha=0.6)
    ax1.plot(df['timestamp'], df['acceleration_y'], label='Acceleration Y', color='tab:green', alpha=0.6)
    ax1.plot(df['timestamp'], df['acceleration_z'], label='Acceleration Z', color='tab:red', alpha=0.6)
    ax1.tick_params(axis='y', labelcolor='tab:blue')

    # Instantiate a second y-axis sharing the same x-axis
    ax2 = ax1.twinx()
    ax2.set_ylabel('Gyroscope', color='tab:orange')
    ax2.plot(df['timestamp'], df['gyroscope_x'], label='Gyroscope X', color='tab:orange', alpha=0.6)
    ax2.plot(df['timestamp'], df['gyroscope_y'], label='Gyroscope Y', color='tab:purple', alpha=0.6)
    ax2.plot(df['timestamp'], df['gyroscope_z'], label='Gyroscope Z', color='tab:brown', alpha=0.6)
    ax2.tick_params(axis='y', labelcolor='tab:orange')

    # Title and legend
    plt.title(f'{activity} Activity Sensor Data')
    fig.tight_layout()  # Adjusts the plot to ensure everything fits without overlapping
    ax1.legend(loc='upper left')
    ax2.legend(loc='upper right')

    # Show plot
    plt.show()


def plot_sensor_data_by_sample(df, activity):
    # Create a figure and a set of subplots
    fig, ax1 = plt.subplots(figsize=(12, 6))

    # Plot acceleration data
    ax1.set_xlabel('Sample')
    ax1.set_ylabel('Acceleration', color='tab:blue')
    ax1.plot(df.index, df['acceleration_x'], label='Acceleration X', color='tab:blue', alpha=0.6)
    ax1.plot(df.index, df['acceleration_y'], label='Acceleration Y', color='tab:green', alpha=0.6)
    ax1.plot(df.index, df['acceleration_z'], label='Acceleration Z', color='tab:red', alpha=0.6)
    ax1.tick_params(axis='y', labelcolor='tab:blue')

    # Instantiate a second y-axis sharing the same x-axis
    ax2 = ax1.twinx()
    ax2.set_ylabel('Gyroscope', color='tab:orange')
    ax2.plot(df.index, df['gyroscope_x'], label='Gyroscope X', color='tab:orange', alpha=0.6)
    ax2.plot(df.index, df['gyroscope_y'], label='Gyroscope Y', color='tab:purple', alpha=0.6)
    ax2.plot(df.index, df['gyroscope_z'], label='Gyroscope Z', color='tab:brown', alpha=0.6)
    ax2.tick_params(axis='y', labelcolor='tab:orange')

    # Title and legend
    plt.title(f'{activity} Activity Sensor Data by Sample')
    fig.tight_layout()  # Adjusts the plot to ensure everything fits without overlapping
    ax1.legend(loc='upper left')
    ax2.legend(loc='upper right')

    # Show plot
    plt.show()

def smooth_window(df,activity):
    df['timestamp'] = pd.to_datetime(df['timestamp'])

    # Sort DataFrame by timestamp (if it's not already sorted)
    df = df.sort_values(by='timestamp')

    # Set timestamp as the index for time-based operations
    #df.set_index('timestamp', inplace=True)

    # Plot original data for comparison
    fig, ax = plt.subplots(figsize=(12, 6))
    ax.plot(df.index, df['acceleration_x'], label='Original Data', color='tab:blue', alpha=0.6)

    # Apply rolling average
    window_size = 5  # Adjust the window size as needed
    df['acceleration_x_smoothed'] = df['acceleration_x'].rolling(window=window_size).mean()

    # Plot smoothed data
    ax.plot(df.index, df['acceleration_x_smoothed'], label=f'Rolling Average (Window Size = {window_size})',
            color='tab:red')

    # Customize the plot
    ax.set_xlabel('Time')
    ax.set_ylabel('Acceleration X')
    ax.legend()
    plt.title('Original and Smoothed Acceleration X {activity} Data')
    plt.show()

def compute_time_domain_features(df, activity):
    # Compute time domain features
    mean_values = df.mean()
    std_dev_values = df.std()
    energy_values = (df**2).sum()  # Energy is the sum of squared values
    spectral_entropy_values = compute_spectral_entropy(df)

    # Print or use the features as needed
    print(f"Time Domain Features for {activity} Activity:")
    print("Mean Values:")
    print(mean_values)
    print("\nStandard Deviation Values:")
    print(std_dev_values)
    print("\nEnergy Values:")
    print(energy_values)
    print("\nSpectral Entropy Values:")
    print(spectral_entropy_values)

def compute_spectral_entropy(df):
    # Compute the power spectral density
    f, Pxx = signal.periodogram(df, axis=0)

    # Normalize the power spectrum
    Pxx /= Pxx.sum(axis=0, keepdims=True)

    # Compute spectral entropy
    spectral_entropy = -(Pxx * np.log2(Pxx)).sum(axis=0)

    return spectral_entropy

def describe_data(df, activity):
    stats = df.describe()
    print("Descriptive Statistics for {activity} Data:")
    print(stats)

for column in pickup_data.columns[:-2]:
    stat, p_value = mannwhitneyu(pickup_data[column], walk_data[column])
    print(f"Mann-Whitney U Test for {column}: U = {stat}, p-value = {p_value}")

def butter_lowpass_filter(data, cutoff_frequency, fs, order=4):
    nyquist = 0.5 * fs
    normal_cutoff = cutoff_frequency / nyquist
    b, a = butter(order, normal_cutoff, btype='low', analog=False)
    y = filtfilt(b, a, data)
    return y

# Example: Apply the filter to acceleration_x column in pickup_data
#pickup_data['acceleration_x_filtered'] = butter_lowpass_filter(pickup_data['acceleration_x'], cutoff_frequency=100, fs=1000)

def plot_sensor_filtered_data(df, activity):
    # Convert timestamp to datetime
    df['timestamp'] = pd.to_datetime(df['timestamp'])
    
    # Plot each column separately on the same graph
    fig, axs = plt.subplots(nrows=2, ncols=3, figsize=(15, 8))

    columns_to_plot = ['acceleration_x', 'acceleration_y', 'acceleration_z', 'gyroscope_x', 'gyroscope_y', 'gyroscope_z']

    for i, ax in enumerate(axs.flatten()):
        col = columns_to_plot[i]
        raw_data = df[col].values
        filtered_data = butter_lowpass_filter(raw_data, cutoff_frequency=100, fs=1000, order=4)

        ax.plot(df.index, raw_data, label='Raw Data', alpha=0.6)
        ax.plot(df.index, filtered_data, label='Filtered Data', alpha=0.8)
        
        ax.set_title(col)
        ax.set_xlabel('Data Index')
        ax.set_ylabel('Value')
        ax.legend()

    # Adjust layout
    plt.tight_layout()
    plt.suptitle(f'{activity} Activity Sensor Data', y=1.05)
    plt.show()

# Plotting for pickup data
#plot_sensor_data(pickup_data, "Pickup")

# Plotting for walk data

#plot_sensor_data(walk_data, "Walk")
# Example usage for pickup data
compute_time_domain_features(pickup_data[['acceleration_x', 'acceleration_y', 'acceleration_z', 'gyroscope_x', 'gyroscope_y', 'gyroscope_z']], "Pickup")

# Example usage for walk data
compute_time_domain_features(walk_data[['acceleration_x', 'acceleration_y', 'acceleration_z', 'gyroscope_x', 'gyroscope_y', 'gyroscope_z']], "Walk")
# Plotting for pickup data by sample
plot_sensor_data_by_sample(pickup_data, "Pickup")

# Plotting for walk data by sample
plot_sensor_data_by_sample(walk_data, "Walk")

smooth_window(pickup_data,"Pickup")
smooth_window(walk_data,"Walk")

describe_data(pickup_data,"pickup")
describe_data(walk_data,"walk")

# Plotting for pickup filtered data
plot_sensor_filtered_data(pickup_data, "Pickup")

# Plotting for walk filtered data
plot_sensor_filtered_data(walk_data, "Walk")
