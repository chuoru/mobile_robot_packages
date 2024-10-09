# flake8: noqa
import numpy as np
import matplotlib.pyplot as plt

# Define cycle-specific velocity profiles
cycle_velocities = [
    [400, -400, 500, -500, 600, -600, 800, -800],  # Cycle 1
    [400, -400, 500, -500, 600, -600, 700, -700],  # Cycle 2
    [430, -430, 530, -530, 630, -630, 730, -730, 800, -800],  # Cycle 3
    [470, -470, 570, -570, 670, -670, 770, -770, 800, -800]   # Cycle 4
]

# Define cycle-specific acceleration profiles
cycle_accelerations = [
    [200, 400, 600, 800],   # Cycle 1
    [3000],              # Cycle 2
    [4000],              # Cycle 3
    [5000]               # Cycle 4
]

# Parameters
time_step = 0.01  # seconds, time resolution of the simulation
hold_time = 0.5  # Time to hold velocity constant before transitioning (in seconds) for cycles with holding
num_repetitions = 3  # Number of repetitions of the 4-cycle set

# Function to generate velocity profiles
def generate_velocity_profile(accelerations, velocities, time_step, hold_time, cycle_with_hold=False):
    velocity_profile = []
    acceleration_profile = []
    time_profile = []
    current_velocity = velocities[0]
    
    # Initialize time at 0
    current_time = 0
    
    for i in range(1, len(velocities)):
        target_velocity = velocities[i]
        velocity_change = target_velocity - current_velocity
        
        # Use the specific accelerations for this cycle
        if i - 1 < len(accelerations):
            accel_value = accelerations[i - 1] * np.sign(velocity_change)
        else:
            accel_value = accelerations[-1] * np.sign(velocity_change)  # Use the last acceleration after the cycle ends
        
        # Calculate the time required to reach the target velocity using the specific acceleration
        required_time = abs(velocity_change) / abs(accel_value)
        
        # Generate time for this segment
        segment_time = np.arange(0, required_time, time_step)
        
        for t in segment_time:
            current_velocity += accel_value * time_step
            
            # Cap the velocity at the target value to avoid overshooting
            if accel_value > 0 and current_velocity > target_velocity:
                current_velocity = target_velocity
            elif accel_value < 0 and current_velocity < target_velocity:
                current_velocity = target_velocity
            
            # Append the values to the profiles
            velocity_profile.append(current_velocity)
            acceleration_profile.append(accel_value if current_velocity != target_velocity else 0)
            time_profile.append(current_time)
            current_time += time_step
        
        # Add hold time if this is a cycle with holding
        if cycle_with_hold:
            hold_segment_time = np.arange(0, hold_time, time_step)
            for _ in hold_segment_time:
                velocity_profile.append(current_velocity)
                acceleration_profile.append(0)  # No acceleration during holding period
                time_profile.append(current_time)
                current_time += time_step
    
    return np.array(velocity_profile), np.array(acceleration_profile), np.array(time_profile)

# Function to repeat the entire 4-cycle set for the desired number of repetitions
def repeat_cycles(num_repetitions):
    full_left_vel = []
    full_right_vel = []
    full_left_accel = []
    full_right_accel = []
    full_time = []

    cycle_time_offset = 0
    
    for _ in range(num_repetitions):
        for cycle_idx in range(4):
            # Generate the cycle with the specific velocity and acceleration profiles
            left_vel_cycle, left_accel_cycle, time_cycle = generate_velocity_profile(
                cycle_accelerations[cycle_idx], cycle_velocities[cycle_idx], time_step, hold_time, cycle_with_hold=(cycle_idx > 0))
            right_vel_cycle, right_accel_cycle, _ = generate_velocity_profile(
                cycle_accelerations[cycle_idx], cycle_velocities[cycle_idx], time_step, hold_time, cycle_with_hold=(cycle_idx > 0))

            # Adjust time for each cycle to follow the previous one
            time_cycle = time_cycle + cycle_time_offset

            # Update time offset for the next repetition
            cycle_time_offset = time_cycle[-1] + time_step

            # Append the current set to the full trajectory
            full_left_vel.append(left_vel_cycle)
            full_right_vel.append(right_vel_cycle)
            full_left_accel.append(left_accel_cycle)
            full_right_accel.append(right_accel_cycle)
            full_time.append(time_cycle)

    # Concatenate the full trajectory
    full_left_vel = np.concatenate(full_left_vel)
    full_right_vel = np.concatenate(full_right_vel)
    full_left_accel = np.concatenate(full_left_accel)
    full_right_accel = np.concatenate(full_right_accel)
    full_time = np.concatenate(full_time)
    
    return full_left_vel, full_right_vel, full_left_accel, full_right_accel, full_time

# Generate the repeated cycles
left_vel, right_vel, left_accel, right_accel, time = repeat_cycles(num_repetitions)

# Plot the combined profiles
fig, axs = plt.subplots(2, 1, figsize=(12, 8))

# Plot accelerations first
axs[0].plot(time, left_accel, label="Left Motor Acceleration Profile", color='black')
axs[0].plot(time, right_accel, label="Right Motor Acceleration Profile", linestyle='dashed', color='grey')
axs[0].set_title("Motor Acceleration Profiles (Repeated Cycles)")
axs[0].set_xlabel("Time [s]")
axs[0].set_ylabel("Acc. [mm/s^2]")
axs[0].grid(True)
axs[0].legend()

# Plot velocities next
axs[1].plot(time, left_vel, label="Left Motor Velocity Profile (Excited)", color='black')
axs[1].plot(time, right_vel, label="Right Motor Velocity Profile (Excited)", linestyle='dashed', color='grey')
axs[1].set_title("Motor Velocity Profiles (Repeated Cycles)")
axs[1].set_xlabel("Time [s]")
axs[1].set_ylabel("Vel. [mm/s]")
axs[1].grid(True)
axs[1].legend()

plt.tight_layout()
plt.show()
