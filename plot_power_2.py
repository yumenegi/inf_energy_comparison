import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import os
import csv
import re # To parse the comment lines

def calculate_energy_from_csv(csv_filepath, workload_start_time, workload_end_time):
    """Calculate energy from CSV file logged from GPU profiler"""
    if not os.path.exists(csv_filepath):
        print(f"Error: CSV file not found at {csv_filepath}")
        return None

    try:
        df = pd.read_csv(
            csv_filepath,
            skipinitialspace=True,
            names=["timestamp", "power_mW"], # Specify expected column names
            header=0           # The first line is the header
        )

        # Convert columns to appropriate types
        df['timestamp'] = pd.to_numeric(df['timestamp'], errors='coerce')
        df['power_mW'] = pd.to_numeric(df['power_mW'], errors='coerce')

        # Drop rows with invalid data after conversion
        df = df.dropna().reset_index(drop=True)

        if df.empty:
             print("Error: No valid numeric data found in the CSV after parsing.")
             return None

        # Ensure timestamps are sorted, although they should be if logged sequentially
        df = df.sort_values(by='timestamp').reset_index(drop=True)

        # --- Subtract Baseline Idle Power ---
        # Use the last power reading as the baseline
        baseline_power_mw = df['power_mW'].iloc[-1]
        print(f"Using baseline idle power from last reading: {baseline_power_mw:.2f} mW")

        # Subtract the baseline from all power readings
        # Ensure the result is not negative using np.maximum
        df['power_mw_adjusted'] = np.maximum(0, df['power_mW'] - baseline_power_mw)


        workload_indices = np.where(
            (df['timestamp'].values >= workload_start_time) & (df['timestamp'].values <= workload_end_time)
        )[0]

        if len(workload_indices) < 2:
            print("Not enough data points found within the specified workload duration for energy calculation.")
            print(f"Time window: [{workload_start_time}, {workload_end_time}]")
            print(f"Found {len(workload_indices)} points.")
            # Return 0.0 if at least one point is found, otherwise None
            return 0.0 if len(workload_indices) > 0 else None

        # Extract timestamps and ADJUSTED power for the workload duration
        workload_timestamps = df['timestamp'].values[workload_indices]
        workload_power_mw_adjusted = df['power_mw_adjusted'].values[workload_indices]

        # Calculate time differences between consecutive points (in seconds)
        time_deltas = np.diff(workload_timestamps)

        # Calculate average ADJUSTED power for each interval using the trapezoidal rule (in mW)
        # For N points, there are N-1 intervals. np.diff gives N-1 deltas.
        average_power_mw_per_interval = (workload_power_mw_adjusted[:-1] + workload_power_mw_adjusted[1:]) / 2.0

        # Convert average power from mW to Watts (divide by 1000)
        average_power_watts_per_interval = average_power_mw_per_interval / 1000.0

        # Calculate energy for each interval (Watts * Seconds)
        energy_per_interval_joules = average_power_watts_per_interval * time_deltas

        # Sum the energy for all intervals within the workload duration
        total_energy_joules = np.sum(energy_per_interval_joules)

        return float(total_energy_joules) # Return as float

    except Exception as e:
        print(f"An error occurred while processing the CSV or calculating energy: {e}")
        return None

def plot_gpu_power_from_csv(csv_filepath="gpu_power_profile.csv", start_time = None, end_time = None, save_path="power_profile_2.png"):
    """
    Loads GPU power data and workload timestamps from a CSV file and plots the 
    power draw over time with markers.
    """
    if not os.path.exists(csv_filepath):
        print(f"Error: CSV file not found at {csv_filepath}")
        print("Please ensure 'run_workload_and_profile.py' was run and generated this file.")
        return

    workload_start_time = start_time
    workload_end_time = end_time
    data_header_row = None
    data_rows = []

    try:
        with open(csv_filepath, 'r') as csvfile:
            reader = csv.reader(csvfile)
            for i, row in enumerate(reader):
                if not row: # Skip empty rows
                    continue

                # Check for metadata comments
                if row[0].strip().startswith('#'):
                    line = row[0].strip()
                    start_match = re.match(r"# Workload Start Time \(abs\): (\d+\.?\d*)", line)
                    end_match = re.match(r"# Workload End Time \(abs\): (\d+\.?\d*)", line)

                    if start_match:
                        workload_start_time = float(start_match.group(1))
                        print(f"Found Workload Start Time (abs): {workload_start_time}")
                    elif end_match:
                        workload_end_time = float(end_match.group(1))
                        print(f"Found Workload End Time (abs): {workload_end_time}")

                # Identify the header row (assuming it's the first non-comment row after metadata)
                elif data_header_row is None:
                    data_header_row = row
                    # Assuming the header is exactly ["timestamp", "power_mW"]
                    if data_header_row != ["timestamp", "power_mW"]:
                        print(f"Warning: Unexpected CSV header: {data_header_row}. Expected ['timestamp', 'power_mW']")
                    # The actual data starts on the next row
                    data_start_row_index = i + 1

                # Collect data rows after the header
                elif data_header_row is not None:
                    data_rows.append(row)

    except Exception as e:
        print(f"Error reading CSV metadata and data: {e}")
        return

    if data_header_row is None or not data_rows:
        print(f"Error: Could not find data header or data rows in {csv_filepath}")
        return

    try:
        df = pd.read_csv(
            csv_filepath,
            comment='#',
            skipinitialspace=True,
            names=["timestamp", "power_mW"], # Specify expected column names
            header=0           # The first non-comment line is the header
        )

        # Convert columns to appropriate types
        df['timestamp'] = pd.to_numeric(df['timestamp'], errors='coerce')
        df['power_mW'] = pd.to_numeric(df['power_mW'], errors='coerce')

        # Drop rows with invalid data after conversion
        df = df.dropna().reset_index(drop=True)

        if df.empty:
             print("Error: No valid numeric data found in the CSV after parsing.")
             return

        timestamps = df['timestamp'].values
        power_mw = df['power_mW'].values

        # Calculate time relative to the first recorded sample
        start_timestamp = timestamps[0]
        relative_times = timestamps - start_timestamp

        plt.figure(figsize=(12, 6))

        # Plot the power data, converting mW to Watts
        plt.plot(relative_times, power_mw / 1000.0, label="GPU Power Draw (W)", alpha=0.8) # Convert mW to W for plot

        # Add vertical markers if start and end times were found
        if workload_start_time is not None and workload_end_time is not None:
            # Calculate relative marker times based on the first sample's timestamp
            relative_start_marker = workload_start_time - start_timestamp
            relative_end_marker = workload_end_time - start_timestamp

            # Add vertical lines
            plt.axvline(relative_start_marker, color='g', linestyle='--', label=f'Workload Start ({relative_start_marker:.2f} s)')
            plt.axvline(relative_end_marker, color='r', linestyle='--', label=f'Workload End ({relative_end_marker:.2f} s)')

        plt.xlabel(f"Time Relative to Profiling Start (s)", fontsize=18)
        plt.ylabel("GPU Power Draw (W)", fontsize=18) # Y-axis is in Watts
        plt.grid(True)
        plt.autoscale(enable=True, axis='x', tight=True)
        plt.legend() # Show the legend for labels

        # Save plot
        plt.savefig(save_path)
        print("Plot saved to power_profile_2.png")

    except Exception as e:
        print(f"An error occurred while processing the CSV or plotting: {e}")
