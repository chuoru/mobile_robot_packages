# flake8: noqa
import os 
import casadi as ca
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

current_directory = os.path.dirname(os.path.abspath(__file__))

data_folder = os.path.join(current_directory, "data")

# Load the data from CSV files
v_data = pd.read_csv(os.path.join('excited_0.2.csv'))  # Adjust the filename as needed
y_data = pd.read_csv(os.path.join('clipped.csv'))  # Adjust the filename as needed

# Extract the relevant columns for the first model
# Adjust based on your CSV column names
v_right_values = v_data['right_vel'].values
y_right_values = y_data['P1'].values  # Adjust based on your CSV column names
t_values = v_data['timestamp'].values

# Calculate 'a' based on consecutive right velocities
sampling_time = 0.05
a_right_values = np.diff(v_right_values) / sampling_time
# To match the length of y_right_values
a_right_values = np.append(a_right_values, a_right_values[-1])

# Define the CasADi variables for the parameters
c1 = ca.MX.sym('c1')
c2 = ca.MX.sym('c2')
c3 = ca.MX.sym('c3')
c4 = ca.MX.sym('c4')
c5 = ca.MX.sym('c5')
c6 = ca.MX.sym('c6')

# Create the model function for right velocity
y_right_model = (c1 * a_right_values**2 +
                 c2 * v_right_values**2 +
                 c3 * a_right_values +
                 c4 * v_right_values +
                 c5 * v_right_values * a_right_values +
                 c6)

# Define the current residuals for right velocity
current_residuals_right = y_right_values - y_right_model

# Previous residuals (for demonstration; typically would be saved from
# previous optimization)
previous_residuals_right = np.zeros_like(y_right_values)  # Initialize to zeros

# Define the cost function for right velocity
cost_right = ca.sum1(current_residuals_right**2) + \
    ca.sum1(previous_residuals_right**2)

# Create an optimization problem for right velocity
nlp_right = {'x': ca.vertcat(c1, c2, c3, c4, c5, c6), 'f': cost_right}

# Set up the solver
solver_right = ca.nlpsol('solver_right', 'ipopt', nlp_right)

# Initial guess for parameters for right velocity
initial_guess_right = np.array([0.1, 0.1, 0.1, 0.1, 0.1, 0.1])

# Solve the optimization problem for right velocity
solution_right = solver_right(x0=initial_guess_right)

# Extract the optimized parameters for right velocity
optimized_params_right = solution_right['x'].full().flatten()
(c1_opt_right, c2_opt_right, c3_opt_right,
 c4_opt_right, c5_opt_right, c6_opt_right) = optimized_params_right

# Calculate the fitted values for right velocity
y_right_fitted = (c1_opt_right * a_right_values**2 +
                  c2_opt_right * v_right_values**2 +
                  c3_opt_right * a_right_values +
                  c4_opt_right * v_right_values +
                  c5_opt_right * v_right_values * a_right_values +
                  c6_opt_right)

# Now perform the same for left velocity
# Extract the relevant columns for the second model
# Adjust based on your CSV column names
v_left_values = v_data['left_vel'].values
y_left_values = y_data['P2'].values  # Adjust based on your CSV column names

# Calculate 'a' based on consecutive left velocities
a_left_values = np.diff(v_left_values) / sampling_time
# To match the length of y_left_values
a_left_values = np.append(a_left_values, a_left_values[-1])

# Define the CasADi variables for the parameters (reuse the same symbols)
# Create the model function for left velocity
y_left_model = (c1 * a_left_values**2 +
                c2 * v_left_values**2 +
                c3 * a_left_values +
                c4 * v_left_values +
                c5 * v_left_values * a_left_values +
                c6)

# Define the current residuals for left velocity
current_residuals_left = y_left_values - y_left_model

# Previous residuals (for demonstration; typically would be saved from
# previous optimization)
previous_residuals_left = np.zeros_like(y_left_values)  # Initialize to zeros

# Define the cost function for left velocity
cost_left = ca.sum1(current_residuals_left**2) + ca.sum1(
    previous_residuals_left**2)

# Create an optimization problem for left velocity
nlp_left = {'x': ca.vertcat(c1, c2, c3, c4, c5, c6), 'f': cost_left}

# Set up the solver
solver_left = ca.nlpsol('solver_left', 'ipopt', nlp_left)

# Initial guess for parameters for left velocity
initial_guess_left = np.array([0.0, 8, 2, 2, 3, 12])

# Solve the optimization problem for left velocity
solution_left = solver_left(x0=initial_guess_left)

# Extract the optimized parameters for left velocity
optimized_params_left = solution_left['x'].full().flatten()
(c1_opt_left, c2_opt_left, c3_opt_left,
 c4_opt_left, c5_opt_left, c6_opt_left) = optimized_params_left

# Calculate the fitted values for left velocity
y_left_fitted = (c1_opt_left * a_left_values**2 +
                 c2_opt_left * v_left_values**2 +
                 c3_opt_left * a_left_values +
                 c4_opt_left * v_left_values +
                 c5_opt_left * v_left_values * a_left_values +
                 c6_opt_left)
print("Optimized parameters for left velocity:", optimized_params_left)
# Plotting the original data and the fitted curves for both velocities
plt.figure(figsize=(12, 6))

# Subplot for right velocity
plt.subplot(2, 1, 1)  # 1 row, 2 columns, 1st subplot
plt.plot(t_values, y_right_values, label='Observed Data (Right)',
         color='blue')  # Original data points
plt.plot(t_values, y_right_fitted, label='Fitted Model (Right)',
         color='red', linewidth=2)  # Fitted curve
plt.xlabel('Timestamp')
plt.ylabel('Motor #1 Power (W)')
plt.title('Nonlinear Least Squares Fitting (Right Velocity)')
plt.legend()
plt.grid()

# Subplot for left velocity
plt.subplot(2, 1, 2)  # 1 row, 2 columns, 2nd subplot
plt.plot(t_values, y_left_values, label='Observed Data (Left)',
         color='green')  # Original data points
plt.plot(t_values, y_left_fitted, label='Fitted Model (Left)',
         color='orange', linewidth=2)  # Fitted curve
plt.xlabel('Timestamp')
plt.ylabel('Motor #2 Power (W)')
plt.title('Nonlinear Least Squares Fitting (Left Velocity)')
plt.legend()
plt.grid()

# Show the plot
plt.tight_layout()  # Adjust layout for better spacing
plt.show()
