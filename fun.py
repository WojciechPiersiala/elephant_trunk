#!/home/wp/Studia/soft_robotics/elephant_trunk/trunk/bin/python
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.animation import FuncAnimation

# Initial manipulator parameters
num_sections = 4
section_lengths = [5, 5, 5, 5]  # Length of each section
curvatures = [0, 0, 0, 0]  # Initial curvature for each section (0 = straight)
angles = [0] * (num_sections + 1)  # Angles between sections
positions = [(0, 0)]  # Start at the origin
selected_section = 0  # Currently selected section

# Function to update the manipulator geometry
def calculate_positions():
    global positions, angles
    positions = [(0, 0)]  # Reset start position
    angles[0] = 0  # Base angle
    for i in range(num_sections):
        x_prev, y_prev = positions[-1]
        curvature = curvatures[i]
        length = section_lengths[i]
        angle = angles[i]

        # Compute new angle based on curvature
        new_angle = angle + curvature * length
        angles[i + 1] = new_angle

        # Compute new position
        dx = length * np.cos(new_angle)
        dy = length * np.sin(new_angle)
        positions.append((x_prev + dx, y_prev + dy))

# Function to update the plot
def update_plot():
    x, y = zip(*positions)
    line.set_data(x, y)
    fig.canvas.draw()

# Function to handle keyboard input
def on_key(event):
    global selected_section
    if event.key == "right":
        curvatures[selected_section] += 0.01  # Increase curvature
    elif event.key == "left":
        curvatures[selected_section] -= 0.01  # Decrease curvature
    elif event.key == "up":
        selected_section = (selected_section + 1) % num_sections  # Next section
    elif event.key == "down":
        selected_section = (selected_section - 1) % num_sections  # Previous section

    calculate_positions()
    update_plot()

# Initial setup of the plot
fig, ax = plt.subplots()
ax.set_xlim(-20, 20)
ax.set_ylim(-20, 20)
line, = ax.plot([], [], 'o-', lw=2)

calculate_positions()
update_plot()

# Connect the keyboard event handler
fig.canvas.mpl_connect("key_press_event", on_key)

# Show the plot
plt.show()
