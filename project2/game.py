from typing import List
import matplotlib.pyplot as plt
import numpy as np

plt.ion()


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

    def get_nn_input(self) -> "Game":
        pass

    def mask_illegal_indexes(self, logits):
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

    # Supports both integer and tuple moves
    def make_move(self, move) -> "HexGame":
        if isinstance(move, int) or isinstance(move, np.int64):
            move = np.unravel_index(move, self.board_state.shape)

        assert len(move) == 2
        # p2 uses a different coordinate system where the board is transposed
        # so the anet always tries to connect top and bottom
        # therefore we need to transpose back the move to the original coordinate system
        if not self.p1_turn:
            move = (move[1], move[0])

        assert self.board_state[move[0], move[1]] == self.empty_encoding

        new_board_state = self.board_state.copy()
        new_board_state[move[0], move[1]] = (
            self.p1_encoding if self.p1_turn else self.p2_encoding
        )
        new_board = HexGame(
            size=self.size,
            last_move=move,
            p1_encoding=self.p1_encoding,
            p2_encoding=self.p2_encoding,
            empty_encoding=self.empty_encoding,
            board_state=new_board_state,
            p1_turn=(not self.p1_turn),
        )
        return new_board

    def get_nn_input(self, num_input_channels=1):
        if num_input_channels == 1:
            if self.p1_turn:
                return self.board_state.copy()

            new_board_state_p1 = np.where(
                self.board_state == self.p1_encoding, self.p2_encoding, 0
            )
            new_board_state_p2 = np.where(
                self.board_state == self.p2_encoding, self.p1_encoding, 0
            )
            new_board_state_empty = np.where(
                self.board_state == self.empty_encoding, self.empty_encoding, 0
            )
            # assert self.empty_encoding + self.empty_encoding == self.empty_encoding
            return new_board_state_empty + new_board_state_p1 + new_board_state_p2

        board_state = self.board_state.copy()
        board_size = self.size
        # Surrond the board with colored hexagons
        # to indicate the two sides to connect
        p1_side = np.array([self.p1_encoding] * (board_size + 4)).reshape(1, -1)
        p2_side = np.array([self.p2_encoding] * board_size).reshape(-1, 1)
        board_state = np.concatenate([p2_side, board_state, p2_side], axis=1)
        board_state = np.concatenate([p2_side, board_state, p2_side], axis=1)
        board_state = np.concatenate([p1_side, board_state, p1_side], axis=0)
        board_state = np.concatenate([p1_side, board_state, p1_side], axis=0)
        new_board_state_p1 = np.where(board_state == self.p1_encoding, 1, 0)
        new_board_state_p2 = np.where(board_state == self.p2_encoding, 1, 0)
        new_board_state_empty = np.where(board_state == self.empty_encoding, 1, 0)

        new_board_state_p2[:2, :2] = 1
        new_board_state_p2[:2, -2:] = 1
        new_board_state_p2[-2:, :2] = 1
        new_board_state_p2[-2:, -2:] = 1

        if self.p1_turn:
            return np.stack(
                [new_board_state_p1, new_board_state_p2, new_board_state_empty]
            )

        # We want the player always to try to connect top and bottom, so we need to transpose the board
        new_board_state_p2 = np.transpose(new_board_state_p2)
        new_board_state_p1 = np.transpose(new_board_state_p1)
        new_board_state_empty = np.transpose(new_board_state_empty)
        return np.stack([new_board_state_p2, new_board_state_p1, new_board_state_empty])

    def visualize_board(
        self,
        ax,
        p1_color="red",
        p2_color="blue",
        face_color="white",
        edge_color="black",
    ):
        ax.clear()
        board_size = self.size

        # Create a hexagon function
        def hexagon(center, size):
            """Generate the vertices of a regular hexagon given a center (x,y), and a size (distance from center to any vertex)."""
            angles = np.linspace(0, 2 * np.pi, 7)
            return np.c_[
                (center[0] + size * np.cos(angles)), (center[1] + size * np.sin(angles))
            ]

        # Create the board
        ax.set_aspect("equal")
        ax.axis("off")

        # Calculate offsets
        x_offset = 1.5
        y_offset = np.sqrt(3)

        board_state = self.board_state.copy()
        # Surrond the board with colored hexagons
        # to indicate the two sides to connect
        p1_side = np.array([self.p1_encoding] * (board_size + 2)).reshape(1, -1)
        p2_side = np.array([self.p2_encoding] * board_size).reshape(-1, 1)
        board_state = np.concatenate([p2_side, board_state, p2_side], axis=1)
        board_state = np.concatenate([p1_side, board_state, p1_side], axis=0)

        board_state = np.rot90(board_state, k=-1)
        # Draw the hexagons
        for x in range(len(board_state)):
            for y in range(len(board_state)):
                center = (x * x_offset, (y - x / 2) * y_offset)
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
        plt.draw()
        plt.pause(0.1)

    def get_legal_moves(self) -> List[tuple[int, int]]:
        if not self.p1_turn:
            return np.argwhere(np.transpose(self.board_state) == self.empty_encoding)
        return np.argwhere(self.board_state == self.empty_encoding)

    def get_successor_states(self) -> List["HexGame"]:
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
                    p1_turn=(not self.p1_turn),
                )
            )
        return successor_states

    def is_terminal(self) -> bool:
        return self.check_winner(self.p1_encoding) or self.check_winner(
            self.p2_encoding
        )

    def check_winner(self, player_encoding) -> bool:
        # p1: top <-> bottom
        # p2: left <-> right
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

    def get_winner(self) -> int:
        if self.check_winner(self.p1_encoding):
            return self.p1_encoding
        if self.check_winner(self.p2_encoding):
            return self.p2_encoding
        return self.empty_encoding

    def neighbor_cells(self, x, y) -> List[tuple[int, int]]:
        return [
            (x - 1, y),
            (x + 1, y),
            (x, y - 1),
            (x, y + 1),
            (x - 1, y + 1),
            (x + 1, y - 1),
        ]

    def bfs(self, x, y, player_encoding) -> bool:
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

    def get_state(self) -> np.ndarray:
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

    def mask_illegal_indexes(self, logits):
        if not self.p1_turn:
            return np.where(
                np.transpose(self.board_state) == self.empty_encoding, logits, 0
            )

        return np.where(self.board_state == self.empty_encoding, logits, 0)

    def __eq__(self, other):
        return np.array_equal(self.board_state, other.board_state)


if __name__ == "__main__":
    # Test is_terminal
    game = [[1, 1, 0], [-1, -1, -1], [0, 0, 0]]
    game = np.array(game)
    hex_game = HexGame(3, board_state=game, last_move=None)
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
    hex_game = HexGame(7, board_state=game, last_move=None)
    print(hex_game.get_legal_moves())
    assert hex_game.is_terminal() == True
    assert hex_game.get_winner() == -1
    game = np.rot90(game, k=1)
    print(game)
    hex_game = HexGame(7, board_state=game, last_move=None)
    assert hex_game.is_terminal() == False
    assert hex_game.get_winner() == 0

    # Test get_nn_input
    game = np.array([[1, 1, 0], [-1, -1, -1], [0, 0, 0]])
    hex_game = HexGame(3, board_state=game, last_move=None)
    assert np.array_equal(hex_game.get_nn_input(), game)
    hex_game = hex_game.make_move((2, 0))
    assert hex_game.board_state[2, 0] == 1
    game[2, 0] = 1
    assert np.array_equal(
        hex_game.get_nn_input(), game * (-1)
    ), f"{hex_game.get_nn_input()} != {game * (-1)}"

    # # Test make_distribution
    from project2.node import Node  # Can't be on top due to import loop

    game = np.array([[0, 0, 1], [0, 0, 0], [0, 0, -1]])
    hex_game = HexGame(3, board_state=game, last_move=None)
    root = Node(parent=None, game_state=hex_game)
    root.visits = 3
    hex_game1 = hex_game.make_move((0, 0))
    hex_game2 = hex_game.make_move((0, 1))
    n1 = Node(parent=root, game_state=hex_game1)
    n1.visits = 1
    n2 = Node(parent=root, game_state=hex_game2)
    n2.visits = 1
    root.add_child(n1)
    root.add_child(n2)
    distribution = hex_game.make_distribution(root.children)
    assert distribution[0, 2] == 0
    assert distribution[2, 2] == 0
    assert distribution[0, 0] == distribution[0, 1]
    assert np.all(
        distribution[i, j] > 0
        for (i, j) in [(0, 0), (0, 1), (1, 0), (1, 1), (1, 2), (2, 1), (2, 2)]
    )

    # Test mask_illegal_indexes
    game = np.array([[0, 0, 1], [0, 0, 0], [0, 0, -1]])
    hex_game = HexGame(3, board_state=game, last_move=None)
    logits = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
    logits = logits / logits.sum()
    masked_logits = hex_game.mask_illegal_indexes(logits)
    assert masked_logits[0, 2] == 0
    assert masked_logits[2, 2] == 0

    # Test make_move
    game = np.array([[0, 0, 1], [0, 0, 0], [0, 0, -1]])
    hex_game = HexGame(3, board_state=game, last_move=None)
    assert hex_game.p1_turn
    hex_game = hex_game.make_move((0, 1))
    assert hex_game.board_state[0, 1] == 1
    assert hex_game.p1_turn == False
    hex_game = hex_game.make_move(4)
    assert hex_game.board_state[1, 1] == -1

    # Test visualize_board
    board = np.array([[0, 0, 1], [-1, 0, 0], [-1, 0, 0]])
    hex_game = HexGame(3, board_state=board, last_move=None)
    fig, ax = plt.subplots(1)
    hex_game.visualize_board(ax)
    plt.show()
    input("Worked?")

    print("All tests passed!")
