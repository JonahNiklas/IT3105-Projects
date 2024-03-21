from typing import List
import matplotlib.pyplot as plt
import numpy as np


class Game:
    def __init__(self):
        raise NotImplementedError(
            "Game is an abstract class and cannot be instantiated"
        )

    def visualize_board(self):
        pass

    def get_legal_moves(self):
        pass

    def get_successor_states(self, player_encoding) -> list["Game"]:
        pass

    def is_terminal(self):
        pass

    def get_winner(self):
        pass

    def make_move(self, move) -> "Game":
        pass

    def __eq__(self, __value: object) -> bool:
        pass


class HexGame(Game):
    def __init__(
        self,
        size,
        last_move,
        p1_encoding=1,
        p2_encoding=-1,
        empty_encoding=0,
        board_state=None,
        p1_turn=True,
    ):
        if board_state is None:
            board_state = np.full((size, size), empty_encoding)
        else:
            assert board_state.shape == (size, size)

        self.size = size
        self.board_state = board_state
        self.p1_encoding = p1_encoding
        self.p2_encoding = p2_encoding
        self.empty_encoding = empty_encoding
        self.p1_turn = p1_turn
        self.last_move = last_move

    def make_move(self, move) -> "HexGame":
        assert len(move) == 2
        assert self.board_state[move[0], move[1]] == self.empty_encoding
        new_board_state = self.board_state.copy()
        new_board = HexGame(
            size=self.size,
            last_move=move,
            p1_encoding=self.p1_encoding,
            p2_encoding=self.p2_encoding,
            empty_encoding=self.empty_encoding,
            board_state=new_board_state,
            p1_turn=self.p1_turn,
        )
        new_board.board_state[move[0], move[1]] = (
            self.p1_encoding if self.p1_turn else self.p2_encoding
        )
        new_board.last_move = move
        new_board.p1_turn = not self.p1_turn
        return new_board

    def visualize_board(
        self, p1_color="red", p2_color="blue", face_color="white", edge_color="black"
    ):

        board_size = self.size

        # Create a hexagon function
        def hexagon(center, size):
            """Generate the vertices of a regular hexagon given a center (x,y), and a size (distance from center to any vertex)."""
            angles = np.linspace(0, 2 * np.pi, 7)
            return np.c_[
                (center[0] + size * np.cos(angles)), (center[1] + size * np.sin(angles))
            ]

        # Create the board
        fig, ax = plt.subplots(1)
        ax.set_aspect("equal")
        ax.axis("off")

        # Calculate offsets
        x_offset = 1.5
        y_offset = np.sqrt(3)

        board_state = np.rot90(self.board_state, k=2)

        # Draw the hexagons
        for x in range(board_size):
            for y in range(board_size):
                center = (x * x_offset, (y + x / 2) * y_offset)
                hex_path = hexagon(center, 1)
                ax.fill(
                    hex_path[:-1, 0],
                    hex_path[:-1, 1],
                    edgecolor=edge_color,
                    facecolor=(
                        face_color
                        if board_state[x, y] == self.empty_encoding
                        else (
                            p1_color
                            if board_state[x, y] == self.p1_encoding
                            else p2_color
                        )
                    ),
                )
                ax.fill(
                    hex_path[-1:, 0],
                    hex_path[-1:, 1],
                    edgecolor=edge_color,
                    facecolor="red",
                )

        plt.gca().invert_xaxis()
        plt.show()

    def get_legal_moves(self):
        return np.argwhere(self.board_state == self.empty_encoding)

    def get_successor_states(self):
        player_encoding = self.p1_encoding if self.p1_turn else self.p2_encoding
        legal_moves = self.get_legal_moves()
        successor_states = []
        for move in legal_moves:
            board_state = self.board_state.copy()
            board_state[move[0], move[1]] = player_encoding
            successor_states.append(
                HexGame(
                    size=self.size,
                    last_move=move,
                    p1_encoding=self.p1_encoding,
                    p2_encoding=self.p2_encoding,
                    empty_encoding=self.empty_encoding,
                    board_state=board_state,
                   p1_turn= not self.p1_turn,
                )
            )
        return successor_states

    def is_terminal(self):
        return self.check_winner(self.p1_encoding) or self.check_winner(
            self.p2_encoding
        )

    def check_winner(self, player_encoding):
        if player_encoding == self.p1_encoding:
            for i in range(self.size):
                if self.board_state[0, i] == player_encoding:
                    if self.bfs(0, i, player_encoding):
                        return True
        else:
            for i in range(self.size):
                if self.board_state[i, 0] == player_encoding:
                    if self.bfs(i, 0, player_encoding):
                        return True
        return False

    def get_winner(self):
        if self.check_winner(self.p1_encoding):
            return self.p1_encoding
        if self.check_winner(self.p2_encoding):
            return self.p2_encoding
        return self.empty_encoding

    def neighbor_cells(self, x, y):
        return [
            (x - 1, y),
            (x + 1, y),
            (x, y - 1),
            (x, y + 1),
            (x - 1, y + 1),
            (x + 1, y - 1),
        ]

    def bfs(self, x, y, player_encoding):
        queue = [(x, y)]
        visited = set()
        while queue:
            x, y = queue.pop(0)
            if x == self.size - 1 and player_encoding == self.p1_encoding:
                return True
            if y == self.size - 1 and player_encoding == self.p2_encoding:
                return True
            for i, j in self.neighbor_cells(x, y):
                if (
                    0 <= i < self.size
                    and 0 <= j < self.size
                    and (i, j) not in visited
                    and self.board_state[i, j] == player_encoding
                ):
                    queue.append((i, j))
                    visited.add((i, j))
        return False

    def get_state(self):
        return self.board_state.copy()

    def make_distribution(self, children: List):
        legal_moves = self.get_legal_moves()
        distribution = np.zeros((self.size, self.size))
        soft_max_sum = np.sum(np.exp([child.visits for child in children])) + (
            len(legal_moves) - len(children)
        )
        for child in children:
            move = child.game_state.last_move
            distribution[move[0], move[1]] = np.exp(child.visits) / soft_max_sum
        for move in legal_moves:
            if distribution[move[0], move[1]] == 0:
                distribution[move[0], move[1]] = 1 / soft_max_sum

        return distribution

    def __eq__(self, other):
        return np.array_equal(self.board_state, other.board_state)


if __name__ == "__main__":
    # Test is_terminal
    game = [[1, 1, 0], [-1, -1, -1], [0, 0, 0]]
    game = np.array(game)
    hex_game = HexGame(3, board_state=game)
    assert hex_game.is_terminal() == True
    assert hex_game.get_winner() == -1

    game = [
        [0, 0, 0, 0, 0, 0, -1],
        [0, 0, 0, 0, 0, -1, 0],
        [0, 0, 0, 0, -1, 0, 0],
        [0, 0, 0, -1, 0, 0, 0],
        [0, 0, -1, 0, 0, 0, 0],
        [0, -1, 0, 0, 0, 0, 0],
        [-1, 0, 0, 0, 0, 0, 0],
    ]
    game = np.array(game)
    hex_game = HexGame(7, board_state=game)
    print(hex_game.get_legal_moves())
    assert hex_game.is_terminal() == True
    assert hex_game.get_winner() == -1

    game = np.rot90(game, k=1)
    print(game)
    hex_game = HexGame(7, board_state=game)
    assert hex_game.is_terminal() == False
    assert hex_game.get_winner() == 0

    print("All tests passed!")
