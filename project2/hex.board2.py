import matplotlib.pyplot as plt
import numpy as np

# Set the size of the board
board_size = 4  # Distance from center to edge

# Create a hexagon function
def hexagon(center, size):
    """Generate the vertices of a regular hexagon given a center (x,y), and a size (distance from center to any vertex)."""
    angles = np.linspace(0, 2*np.pi, 7)
    return np.c_[(center[0] + size * np.cos(angles)), (center[1] + size * np.sin(angles))]

# Create the board
fig, ax = plt.subplots(1)
ax.set_aspect('equal')
ax.axis('off')

# Calculate offsets
x_offset = 1.5
y_offset = np.sqrt(3)

# Draw the hexagons
for x in range(-board_size, board_size + 1):
    for y in range(-board_size, board_size + 1):
        # Skip tiles that are outside the board
        if abs(x + y) > board_size:
            continue
        center = (x * x_offset, (y + x/2) * y_offset)
        hex_path = hexagon(center, 1)
        ax.fill(hex_path[:, 0], hex_path[:, 1], edgecolor='black', fill=False)

# Adjust the plot limits and show the board
ax.set_xlim(-board_size * 2, board_size * 2)
ax.set_ylim(-board_size * 2, board_size * 2)
plt.show()
