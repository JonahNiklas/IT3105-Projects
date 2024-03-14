from project2.game import Game
from project2.globals import MCTS_C, MCTS_ROLLOUT_EPSILON, NUMBER_OF_SIMULATIONS
import numpy as np

class Node():
    def __init__(self, parent, game_state: Game) -> None:
        self.parent = parent
        parent.add_child(self)
        self.game_state = game_state
        self.children = []
        self.visits = 0
        self.Q = 0
    
    def is_leaf(self):
        return len(self.children) == 0
    
    def is_root(self):
        return self.parent is None
    
    def add_child(self, node):
        self.children.append(node)

    def set_to_root(self):
        self.parent = None


class MCTS():
    def __init__(self, ANET) -> None:
        self.ANET = ANET    
        self.root = Node(None) 
        
    def tree_policy(self):
        node = self.root
        while not node.is_leaf():
            parent = node
            node = self.best_uct(node)
        parent.add_child(node)
        return node
    
    def U(self, parent, child):
        return MCTS_C * np.sqrt(np.log(parent.visits) / (1 + child.visits))
    
    def Q(self, parent, child):
        return child.Q
    
    def uct(self, parent, child):
        assert child in parent.children
        return self.Q(parent, child) + self.U(parent, child)
    
    def best_uct(self, node: Node):
        possible_next_states = node.game_state.get_successor_states()
        possible_next_states = [Node(node, state) for state in possible_next_states]
        uct_values = [(child, self.uct(node, child)) for child in node.children]
        return node.argmax(uct_values, key=lambda x: x[1])[0]
    
    def rollout(self, game_state: Game):
        while not game_state.is_terminal():
            logits = self.ANET(game_state)
            free_spots = np.where(game_state.board_state == 0)
            logits[~free_spots] = 0
            # Renormalize
            logits = logits / logits.sum()
            if np.random.rand() < MCTS_ROLLOUT_EPSILON:
                move = np.random.choice(game_state.get_legal_moves())
            else:
                move = np.argmax(logits)
            game_state = game_state.make_move(move)

        return game_state.value
    
    def backpropagate(self, node, value):
        while node is not None:
            node.visits += 1
            node.value += value
            node = node.parent

    def search(self, state: Game):
        for i in range(NUMBER_OF_SIMULATIONS):
            node = self.tree_policy(state)
            value = self.rollout(node.game_state)
            self.backpropagate(node, value)
            self.root.add_child(node)
        return self.best_child(self.root)
        
    def new_root(self,node):
        node.set_to_root()
        self.root = node