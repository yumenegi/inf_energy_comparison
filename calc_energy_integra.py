import pandas as pd
import numpy as np
import sys

def calculate_energy(file_path, start_time, end_time):
    """
    Calculates the energy consumed from power data in a CSV file
    between a specified start and end time, subtracting an estimated
    idle power at the 0-second mark.
    """
    try:
        df = pd.read_csv(file_path, header=0, skiprows=[1])
        print(f"Successfully loaded data from '{file_path}'")

    except FileNotFoundError:
        print(f"Error: File not found at '{file_path}'")
        return None
    except Exception as e:
        print(e)
        return None

    required_cols = ['x-axis', 'P2']
    if not all(col in df.columns for col in required_cols):
        missing = [col for col in required_cols if col not in df.columns]
        print(f"Error: Missing required columns: {missing}")
        print("Please ensure your CSV file has 'x-axis' and 'P2' columns.")
        return None

    df['x-axis'] = pd.to_numeric(df['x-axis'], errors='coerce')
    df['P2'] = pd.to_numeric(df['P2'], errors='coerce')

    df.dropna(subset=['x-axis', 'P2'], inplace=True)
    df.sort_values(by='x-axis', inplace=True)

    # Use 0 second as baseline
    if 0.0 not in df['x-axis'].values:
        # If 0.0 is not exactly present, find the closest time point
        closest_zero_index = (df['x-axis'] - 0.0).abs().idxmin()
        idle_power_time = df.loc[closest_zero_index, 'x-axis']
        idle_power = df.loc[closest_zero_index, 'P2']
        print(f"Using power at closest time point {idle_power_time:.4f}s as idle power: {idle_power:.4f}W")
    else:
        # If 0.0 is exactly present, use the power at 0.0s
        idle_power = df[df['x-axis'] == 0.0]['P2'].iloc[0]
        print(f"Using power at 0.0s as idle power: {idle_power:.4f}W")

    # Filter data for the specified time range
    df_filtered = df[(df['x-axis'] >= start_time) & (df['x-axis'] <= end_time)].copy()

    # Check if there is any data within the specified time range
    if df_filtered.empty:
        print(f"No data found between {start_time}s and {end_time}s.")
        return 0.0 # Return 0 energy if no data in range

    # Subtract the idle power from the power values in the filtered data
    df_filtered['P2_adjusted'] = df_filtered['P2'] - idle_power

    # Extract the time and adjusted power data for integration
    time_points = df_filtered['x-axis'].values
    power_values_adjusted = df_filtered['P2_adjusted'].values

    # Integrate since we are not sure of the sampling rate
    energy_joules = np.trapz(power_values_adjusted, time_points)

    return energy_joules

if __name__ == "__main__":
    csv_file = "logs\power\pi_tpu\IntegraVision_1.csv"

    try:
        start_time_input = input("Enter the start time in seconds: ")
        start_time = float(start_time_input)

        end_time_input = input("Enter the end time in seconds : ")
        end_time = float(end_time_input)

        num_inf_input = input("Enter the number of inferences: ")
        num_inf = int(num_inf_input)

    except ValueError:
        print("Invalid input. Please enter valid numbers for start and end times.")
        sys.exit(1) # Exit the script if input is invalid

    # Ensure start time is not greater than end time
    if start_time > end_time:
        print("Error: Start time cannot be greater than end time.")
        sys.exit(1)

    # Calculate the energy
    energy = calculate_energy(csv_file, start_time, end_time)

    # Print the result if calculation was successful
    if energy is not None:
        print(f"\nCalculated energy consumed between {start_time}s and {end_time}s:")
        print(f"{energy:.4f}J")
        print(f"Energy consumed per inference:")
        print(f"{(energy / num_inf * 1000):.4f}mJ")
        
