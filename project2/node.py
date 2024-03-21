
from project2.game import Game


class Node():
    def __init__(self, parent: 'Node', game_state: 'Game') -> None:
        self.parent = parent
        if parent is not None:
            parent.add_child(self)
        self.game_state = game_state
        self.children = []
        self.visits = 0
        self.value = 0
    
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
