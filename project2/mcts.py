from project2.game import Game
from project2.globals import MCTS_C, MCTS_ROLLOUT_EPSILON, NUMBER_OF_SIMULATIONS
import numpy as np

class Node():
    def __init__(self, parent, game_state: Game) -> None:
        self.parent = parent
        if parent is not None:
            parent.add_child(self)
        self.game_state = game_state
        self.children = []
        self.visits = 0
        self.value = 0
        self.Q = 0
    
    def is_leaf(self):
        return len(self.children) == 0
    
    def is_root(self):
        return self.parent is None
    
    def add_child(self, node):
        self.children.append(node)

    def set_to_root(self):
        self.parent = None
    
    def __eq__(self, other):
        return self.game_state == other.game_state


class MCTS():
    def __init__(self, ANET,game_state: Game) -> None:
        self.ANET = ANET
        self.root = Node(None, game_state)
        
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
        if parent.game_state.p1_turn:
            return self.Q(parent, child) + self.U(parent, child)
        return self.Q(parent, child) - self.U(parent, child)
    
    def best_uct(self, parent: Node):
        unseen_value = self.U(parent, Node(None, None))
        uct_values = [(child, self.uct(parent, child)) for child in parent.children]+[(None, unseen_value)]
        if parent.game_state.p1_turn:
            best_child, best_child_uct = np.argmax(uct_values, key=lambda x: x[1])
        else:
            best_child, best_child_uct = np.argmin(uct_values, key=lambda x: x[1])   
        
        if best_child is not None:
            return best_child
        
        # Select random non-seen successor not in node.children
        possible_next_states = parent.game_state.get_successor_states()
        random_successor_state = np.random.choice(possible_next_states)
        while Node(None, random_successor_state) in parent.children:
            possible_next_states.remove(random_successor_state)
            random_successor_state = np.random.choice(possible_next_states)
        return Node(parent, random_successor_state)
        
    
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
        
        best_actual_move = self.best_child(self.root)
        distribution = self.root.game_state.make_distribution(self.root.children)
        self.new_root(best_actual_move)
        return best_actual_move.game_state, distribution
    
    def best_child(self, node):
        return np.argmax([(child.visits, child) for child in node.children], key=lambda x: x[0])[1]
        
    def new_root(self,node):
        node.set_to_root()
        self.root = node