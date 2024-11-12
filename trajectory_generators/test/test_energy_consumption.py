import pandas as pd
import numpy as np

# Optimized parameters from previous fit
c1_opt = 0.1  # Replace with your optimized c1 value
c2_opt = 0.1  # Replace with your optimized c2 value
c3_opt = 0.1  # Replace with your optimized c3 value
c4_opt = 0.1  # Replace with your optimized c4 value
c5_opt = 0.1  # Replace with your optimized c5 value
c6_opt = 0.1  # Replace with your optimized c6 value

# Load the new CSV file containing 'v' data
new_data = pd.read_csv('new_data.csv')  # Adjust the filename as needed

# Extract the velocity values (assuming the column is named 'v')
v_values = new_data['v'].values  # Adjust based on your CSV column name

# Calculate 'a' based on consecutive v values and the sampling time
sampling_time = 0.05  # Assuming the same sampling time
a_values = np.diff(v_values) / sampling_time
a_values = np.append(a_values, a_values[-1])  # To match the length of v_values

# Compute the predicted y-values using the optimized parameters
y_predicted = (c1_opt * a_values**2 + 
               c2_opt * v_values**2 + 
               c3_opt * a_values + 
               c4_opt * v_values + 
               c5_opt * v_values * a_values + 
               c6_opt)

# Sum of the predicted y-values
y_sum = np.sum(y_predicted)

# Output the result
print(f"Sum of predicted y-values: {y_sum}")