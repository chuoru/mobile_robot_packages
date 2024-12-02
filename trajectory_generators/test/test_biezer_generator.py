import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import interp1d


def bezier_curve(P0, P1, P2, P3, n_points=100):
    t = np.linspace(0, 1, n_points).reshape(-1, 1)
    curve = (1-t)**3 * P0 + 3*(1-t)**2*t * P1 + 3*(1-t)*t**2 * P2 + t**3 * P3
    return curve


def interpolate_trajectory(trajectory, desired_velocity):
    distances = np.sqrt(np.sum(np.diff(trajectory, axis=0)**2, axis=1))
    cumulative_arc_length = np.insert(np.cumsum(distances), 0, 0)
    total_length = cumulative_arc_length[-1]
    # Total time based on desired velocity
    total_time = total_length / desired_velocity
    time_steps = cumulative_arc_length / \
        desired_velocity  # Time stamps for each arc length

    interp_x = interp1d(time_steps, trajectory[:, 0], kind='linear')
    interp_y = interp1d(time_steps, trajectory[:, 1], kind='linear')

    # Uniform time steps for interpolation
    new_time = np.arange(0, total_time, 0.05)
    interpolated_x = interp_x(new_time)
    interpolated_y = interp_y(new_time)

    return np.column_stack((interpolated_x, interpolated_y)), new_time


def calculate_velocity(trajectory, t):
    dt = np.diff(t)
    dx = np.diff(trajectory[:, 0]) / dt
    dy = np.diff(trajectory[:, 1]) / dt

    # Linear velocity
    v = np.sqrt(dx**2 + dy**2)

    # Angular velocity
    ddx = np.diff(dx) / dt[:-1]
    ddy = np.diff(dy) / dt[:-1]
    omega = (ddy * dx[:-1] - ddx * dy[:-1]) / (dx[:-1]**2 + dy[:-1]**2)

    # Ensure same length by aligning with t[:-2]
    t_velocity = t[:-2]
    return t_velocity, v[:-1], omega


# Define control points for the segments
P0_1, P1_1, P2_1, P3_1 = np.array([0, 0]), np.array(
    [0.5, 0]), np.array([4, 0]), np.array([4.5, 0.5])
P0_2, P1_2, P2_2, P3_2 = np.array([4.5, 0.5]), np.array(
    [5, 1.0]), np.array([5, 4.5]), np.array([5, 5])

# Generate curves
curve1 = bezier_curve(P0_1, P1_1, P2_1, P3_1, n_points=100)
curve2 = bezier_curve(P0_2, P1_2, P2_2, P3_2, n_points=100)
trajectory = np.vstack((curve1, curve2))

# Adjust linear velocity
desired_velocity = 0.5  # Adjustable linear velocity (m/s)
interpolated_trajectory, time_stamps = interpolate_trajectory(
    trajectory, desired_velocity)

# Calculate velocities
t_velocity, linear_velocity, angular_velocity = calculate_velocity(
    interpolated_trajectory, time_stamps)

# Save velocities to u
u = np.column_stack((t_velocity, linear_velocity, angular_velocity))

# Plot velocities
figure, ax = plt.subplots()
figure.set_size_inches(10, 6)
ax.plot(t_velocity, linear_velocity,
        label="Linear Velocity (v)", color="blue")
ax.plot(t_velocity, angular_velocity,
        label="Angular Velocity (ω)", color="orange")
ax.set_xlabel("Time (s)")
ax.set_ylabel("Velocity (m/s, rad/s)")
ax.set_title("Velocity Profile")
ax.legend()
ax.grid()

_, ax1 = plt.subplots()
ax1.plot(interpolated_trajectory[:, 0],
         interpolated_trajectory[:, 1], label="Trajectory", color="green")
ax1.set_aspect('equal', 'box')
ax1.set_xlabel("X (m)")
ax1.set_ylabel("Y (m)")
ax1.set_title("Trajectory")
ax1.legend()
ax1.grid()

plt.show()
