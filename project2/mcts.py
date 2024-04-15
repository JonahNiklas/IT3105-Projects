from project2.game import Game
from project2.globals import MCTS_C, MCTS_ROLLOUT_EPSILON, NUMBER_OF_SIMULATIONS, VERBOSE
import numpy as np
from project2.neural_network import NeuralNetwork, device
from project2.node import Node
import torch

np.seterr(all="raise")


class MCTS:
    def __init__(self, ANET: NeuralNetwork, game_state: Game) -> None:
        self.ANET = ANET
        self.root = Node(None, game_state)
        self.root.visits = 1

    def tree_policy(self):
        node = self.root
        assert not node.game_state.is_terminal()
        while not node.is_leaf():
            node = self.best_uct(node)
            if node.game_state.is_terminal():
                return node
        if node.visits > 0:
            node = self.best_uct(node)
        return node

    def U(self, parent: Node, child: Node):
        return MCTS_C * np.sqrt(np.log(parent.visits) / (1 + child.visits))

    def Q(self, parent: Node, child: Node):
        return (child.value / child.visits) if child.visits > 0 else 0

    def uct(self, parent: Node, child: Node):
        if parent.game_state.p1_turn:
            return self.Q(parent, child) + self.U(parent, child)
        return self.Q(parent, child) - self.U(parent, child)

    def best_uct(self, parent: Node):
        # Need to know if all children have been expanded
        num_legal_moves = len(parent.game_state.get_legal_moves())
        unseen_value = self.uct(parent, Node(None, None))
        uct_values = [(child, self.uct(parent, child))
                      for child in parent.children]
        # If all children have been expanded, dont add unseen_value
        if len(uct_values) < num_legal_moves:
            uct_values = uct_values + [(None, unseen_value)]
        if parent.game_state.p1_turn:
            best_child, best_child_uct = max(uct_values, key=lambda x: x[1])
        else:
            best_child, best_child_uct = min(uct_values, key=lambda x: x[1])

        # Best child is an explored node
        if best_child is not None:
            if VERBOSE:
                print("Selecting previously explored node")
            return best_child

        if VERBOSE:
            print("Selecting unseen node")
        # Explore unseen node
        successor_states = parent.game_state.get_successor_states()
        # Select random non-seen successor not in node.children
        possible_next_states = [
            state
            for state in successor_states
            if Node(None, state) not in parent.children
        ]
        assert len(possible_next_states) > 0
        return Node(parent, np.random.choice(possible_next_states))

    def rollout(self, node: Node):
        if VERBOSE:
            print(f"Rollout from {node}")
        assert node.parent.visits > 0
        game_state = node.game_state
        while not game_state.is_terminal():
            # if opponents turn, play the best move for the opponent
            input = game_state.get_nn_input(self.ANET.num_input_channels)
            input = torch.tensor(
                input, dtype=torch.float32).unsqueeze(0).to(device)
            logits = self.ANET(input)
            logits = logits.detach().cpu().numpy().squeeze()
            # Mask out illegal moves
            logits = game_state.mask_illegal_indexes(logits)
            if logits.sum() == 0:
                logits = game_state.mask_illegal_indexes(np.ones_like(logits))
                print("All legal moves were masked, choosing first move.")
            # Renormalize
            # logits = logits / logits.sum()
            if np.random.rand() < MCTS_ROLLOUT_EPSILON:
                legal_moves = game_state.get_legal_moves()
                move = legal_moves[np.random.choice(len(legal_moves))]
            else:
                move = np.unravel_index(np.argmax(logits), logits.shape)
            game_state = game_state.make_move(move)
        return game_state.get_winner()

    def backpropagate(self, node: Node, value: int):
        while node is not None:
            node.visits += 1
            node.value += value
            node = node.parent

    def search(self) -> tuple[Game, np.ndarray]:
        for i in range(1, NUMBER_OF_SIMULATIONS + 1):
            node = self.tree_policy()
            result = self.rollout(node)
            self.backpropagate(node, result)

        best_actual_move = self.best_child(self.root)
        distribution = self.root.game_state.make_distribution(
            self.root.children)
        self.new_root(best_actual_move)
        return best_actual_move.game_state, distribution

    def best_child(self, node: Node) -> Node:
        return max(node.children, key=lambda child: child.visits)

    def new_root(self, node: Node):
        node.set_to_root()
        self.root = node


if __name__ == "__main__":
    board = [[1, 0, 0], [0, 1, 0], [0, 0, 0]]
    board = np.array(board)
    logits = np.array([[0.1, 0.2, 0.3], [0.4, 0.5, 0.6], [0.7, 0.8, 0.9]])
    logits = np.where(board == 0.0, logits, 0)
    assert board.shape == (3, 3)
    print()
    print(logits)
    assert np.array_equal(
        logits, np.array([[0.0, 0.2, 0.3], [0.4, 0.0, 0.6], [0.7, 0.8, 0.9]])
    )

    print("All tests passed!")
