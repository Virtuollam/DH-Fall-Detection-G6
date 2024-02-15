import pandas as pd
import matplotlib.pyplot as plt

def plot_sensor_data_index_based(df, title, filename):
    """
    Plots the time series data of acceleration and gyroscope from the dataframe against the index (number of measurements).
    Acceleration data is plotted with solid lines on the primary y-axis.
    Gyroscope data is plotted with dotted lines on the secondary y-axis.
    """
    # Creating the plot
    fig, ax1 = plt.subplots(figsize=(12, 6))

    # Plotting acceleration data on the primary y-axis
    ax1.set_title(title)
    ax1.plot(df.index, df['acceleration_x'], label='Acceleration X', linestyle='-', color='red')
    ax1.plot(df.index, df['acceleration_y'], label='Acceleration Y', linestyle='-', color='green')
    ax1.plot(df.index, df['acceleration_z'], label='Acceleration Z', linestyle='-', color='blue')
    ax1.set_xlabel('Measurement Number')
    ax1.set_ylabel('Acceleration Readings', color='black')
    ax1.tick_params(axis='y', labelcolor='black')
    ax1.legend(loc='upper left')
    ax1.grid(True)

    # Creating the secondary y-axis for gyroscope data
    ax2 = ax1.twinx()
    ax2.plot(df.index, df['gyroscope_x'], label='Gyroscope X', linestyle=':', color='orange')
    ax2.plot(df.index, df['gyroscope_y'], label='Gyroscope Y', linestyle=':', color='purple')
    ax2.plot(df.index, df['gyroscope_z'], label='Gyroscope Z', linestyle=':', color='brown')
    ax2.set_ylabel('Gyroscope Readings', color='black')
    ax2.tick_params(axis='y', labelcolor='black')
    ax2.legend(loc='upper right')

    plt.tight_layout()
    plt.savefig(filename)
    plt.close()

# Replace these paths with the paths to your actual files
pickup_data_path = 'pickup_data1_20240124_110706.csv'
walk_data_path = 'walk_data1_20240124_103004.csv'

# Load the CSV files into pandas dataframes
pickup_data = pd.read_csv(pickup_data_path)
walk_data = pd.read_csv(walk_data_path)

# Plotting for pickup data with index-based plot
plot_sensor_data_index_based(pickup_data, 'Pickup Activity Sensor Data - Index Based', 'pickup.jpg')

# Plotting for walk data with index-based plot
plot_sensor_data_index_based(walk_data, 'Walk Activity Sensor Data - Index Based', 'walk.jpg')


