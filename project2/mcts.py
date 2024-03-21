from project2.game import Game
from project2.globals import MCTS_C, MCTS_ROLLOUT_EPSILON, NUMBER_OF_SIMULATIONS
import numpy as np
from project2.neural_network import NeuralNetwork
from project2.node import Node
import torch


class MCTS:
    def __init__(self, ANET: NeuralNetwork, game_state: Game) -> None:
        self.ANET = ANET
        self.root = Node(None, game_state)

    def tree_policy(self):
        node = self.root
        while (not node.is_leaf()) or (node == self.root):
            parent = node
            node = self.best_uct(node)
        return node

    def U(self, parent: Node, child: Node):
        return MCTS_C * np.sqrt(np.log(parent.visits) / (1 + child.visits))

    def Q(self, parent: Node, child: Node):
        return child.value / child.visits if child.visits > 0 else 0

    def uct(self, parent, child):
        if parent.game_state.p1_turn:
            return self.Q(parent, child) + self.U(parent, child)
        return self.Q(parent, child) - self.U(parent, child)

    def best_uct(self, parent: Node):
        unseen_value = self.uct(parent, Node(None, None))
        uct_values = [(child, self.uct(parent, child)) for child in parent.children] + [
            (None, unseen_value)
        ]
        if parent.game_state.p1_turn:
            best_child, best_child_uct = max(uct_values, key=lambda x: x[1])
        else:
            best_child, best_child_uct = min(uct_values, key=lambda x: x[1])

        if best_child is not None:
            return best_child

        # Select random non-seen successor not in node.children
        possible_next_states = parent.game_state.get_successor_states()
        random_successor_state = np.random.choice(possible_next_states)
        while Node(None, random_successor_state) in parent.children:
            possible_next_states.remove(random_successor_state)
            random_successor_state = np.random.choice(possible_next_states)
        return Node(parent, random_successor_state)

    def rollout(self, leaf_node: Node):
        game_state = leaf_node.game_state
        while not game_state.is_terminal():
            input = game_state.get_state()
            input = torch.tensor(input, dtype=torch.float32).unsqueeze(0)
            logits = self.ANET(input)
            logits = logits.detach().numpy().squeeze()
            logits = np.where(game_state.board_state == 0, logits, 0)
            # Renormalize
            logits = logits / logits.sum()
            if np.random.rand() < MCTS_ROLLOUT_EPSILON:
                legal_moves = game_state.get_legal_moves()
                move = legal_moves[np.random.choice(len(legal_moves))]
            else:
                move = np.unravel_index(np.argmax(logits), logits.shape)
            game_state = game_state.make_move(move)
        return game_state.get_winner()

    def backpropagate(self, node, value):
        while node is not None:
            node.visits += 1
            node.value += value
            node = node.parent

    def search(self):
        for i in range(1, NUMBER_OF_SIMULATIONS + 1):
            print(f"Simulation {i}")
            node = self.tree_policy()
            result = self.rollout(node)
            self.backpropagate(node, result)

        best_actual_move = self.best_child(self.root)
        distribution = self.root.game_state.make_distribution(self.root.children)
        self.new_root(best_actual_move)
        return best_actual_move.game_state, distribution

    def best_child(self, node):
        return max(
            [(child.visits, child) for child in node.children], key=lambda x: x[0]
        )[1]

    def new_root(self, node):
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
