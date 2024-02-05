import matplotlib
matplotlib.use('TkAgg')
import pandas as pd
import matplotlib.pyplot as plt

# Load the CSV files
file_path_pickup = "pickup_data1_20240124_110706.csv"
file_path_walk = "walk_data1_20240124_103004.csv"

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

# Plotting for pickup data
plot_sensor_data(pickup_data, "Pickup")

# Plotting for walk data

plot_sensor_data(walk_data, "Walk")

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

# Plotting for pickup data by sample
plot_sensor_data_by_sample(pickup_data, "Pickup")

# Plotting for walk data by sample
plot_sensor_data_by_sample(walk_data, "Walk")
