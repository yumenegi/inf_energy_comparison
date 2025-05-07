import pandas as pd
import matplotlib.pyplot as plt
import sys

file_path = "logs\power\pi_tpu\IntegraVision_1.csv"

df = pd.read_csv(file_path, header=0, skiprows=[1])
print(f"Successfully loaded data from '{file_path}'")
print("DataFrame head:")
print(df.head())

df['x-axis'] = pd.to_numeric(df.get('x-axis'), errors='coerce')
df['V2'] = pd.to_numeric(df.get('V2'), errors='coerce')
df['I2'] = pd.to_numeric(df.get('I2'), errors='coerce')
df['P2'] = pd.to_numeric(df.get('P2'), errors='coerce')

df.dropna(inplace=True)

# Try to get units
with open(file_path, 'r') as f:
    header_line = f.readline().strip()
    units_line = f.readline().strip()
    units = units_line.split(',')
    header = header_line.split(',')
    x_axis_unit = units[header.index('x-axis')] if 'x-axis' in header and header.index('x-axis') < len(units) else 'Unit Unknown'

try:
    start_time_input = input("Enter the start time in seconds: ")
    start_time = float(start_time_input)

    end_time_input = input("Enter the end time in seconds: ")
    end_time = float(end_time_input)

except ValueError:
    print("Invalid time")
    sys.exit(1) # Exit the script if input is invalid

# Ensure start time is not greater than end time
if start_time > end_time:
    print("Error: Start time cannot be greater than end time.")
    sys.exit(1)

df_filtered = df[(df['x-axis'] >= start_time) & (df['x-axis'] <= end_time)].copy()

if df_filtered.empty:
    print(f"No data found between {start_time}s and {end_time}s.")
    sys.exit(0) # Exit gracefully if no data in range

plt.figure(figsize=(12, 6)) # Set the figure size for the second plot
plt.plot(df_filtered['x-axis'], df_filtered['P2'], label='Power (P2)', color='green')
plt.xlabel(f"{df.columns[0]} ({x_axis_unit})", fontsize=18) # x-axis label with unit
plt.ylabel('Power (Watt)', fontsize=18) # Specific y-label for Power
plt.legend()
plt.grid(True)

plt.show()
