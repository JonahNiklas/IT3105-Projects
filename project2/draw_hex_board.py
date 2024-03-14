import matplotlib.pyplot as plt
import numpy as np


def draw_hex_board(board, p1_color="red", p2_color="blue",face_color="white",edge_color="black", p1_encoding=1,empty_encoding=0):
    board_size = board.shape[0]
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

    board_state = np.rot90(board, k=2)

    # Draw the hexagons
    for x in range(board_size):
        for y in range(board_size):
            center = (x * x_offset, (y + x/2) * y_offset)
            hex_path = hexagon(center, 1)
            ax.fill(hex_path[:, 0], hex_path[:, 1], edgecolor=edge_color, facecolor=face_color if board_state[x, y] == empty_encoding else p1_color if board_state[x, y] == p1_encoding else p2_color)

    # Adjust the plot limits and show the board
    # ax.set_xlim(-board_size * 2, board_size * 2)
    # ax.set_ylim(-board_size * 2, board_size * 2)
    # Adjust the plot limits and show the board
    plt.gca().invert_xaxis()
    plt.show()


dummy_board=np.zeros((5, 5))
dummy_board[0, 0] = 1
dummy_board[1, 1] = 2
draw_hex_board(board=dummy_board)
