# flake8: noqa
import os
import pandas as pd
import matplotlib.pyplot as plt

current_directory = os.path.dirname(os.path.abspath(__file__))

data_folder = os.path.join(current_directory, "data")

# Replace 'your_file.csv' with the actual file path
csv_file = os.path.join(data_folder, 'raw.csv')

# Read the CSV file into a DataFrame
df = pd.read_csv(csv_file)


# Convert Laptime (hh:mm:ss) to milliseconds
def laptime_to_ms(laptime_str):
    h, m, s = map(int, laptime_str.split(':'))
    return (h * 3600 + m * 60 + s) * 1000


# Create a new column 'Total_Laptime(ms)' by combining 'Laptime'
# and 'Laptime(ms)'
df['Total_Laptime(ms)'] = df.apply(lambda row: laptime_to_ms(
    row['Laptime']) + row['Laptime(ms)'], axis=1)

# Clip the data to include only rows where 'Total_Laptime(ms)' is between 4000
# and 205000
df_clipped = df[(df['Total_Laptime(ms)'] >= 40000) & (
    df['Total_Laptime(ms)'] <= 205000)]

df_clipped['Total_Laptime(ms)'] = df_clipped['Total_Laptime(ms)'] - 40000

df_clipped.to_csv(os.path.join(data_folder, 'clipped.csv'), index=False)
# Extract the relevant data
lap_data = df_clipped[['Total_Laptime(ms)', 'P1', 'P2']]
# Plot P1 vs Total_Laptime(ms)
plt.figure(figsize=(10, 5))

# Plot for P1
plt.subplot(2, 1, 1)
plt.plot(lap_data['Total_Laptime(ms)'], lap_data['P1'], color='grey')
plt.title('Motor #1 Power vs Time (ms)')
plt.xlabel('Time (ms)')
plt.ylabel('Motor #1 Power (W)')

# Plot for P2
plt.subplot(2, 1, 2)
plt.plot(lap_data['Total_Laptime(ms)'], lap_data['P2'], color='grey')
plt.title('Motor #2 Power vs Time (ms)')
plt.xlabel('Time (ms)')
plt.ylabel('Motor #2 Power (W)')

# Adjust layout and show the plots
plt.tight_layout()
plt.show()
