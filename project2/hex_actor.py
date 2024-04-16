
import numpy as np
import torch
from project2.game import HexGame
from project2.neural_network import ConvNetwork

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class HexActor():
    def __init__(self, network_filepath="./saved_networks/conv7x7-200-sims-2000-e_anet_100.pt", net=ConvNetwork(), size=7):
        self.net = net
        self.net.load_state_dict(torch.load(
            network_filepath, map_location=device))
        self.size = size

    def get_action(self, game_state):
        p1_turn = game_state[0] == 1
        game_state_1d = np.array(game_state[1:])
        game_state_2d = game_state_1d.reshape(self.size, self.size)
        game = HexGame(board_state=game_state_2d, p1_encoding=1,
                       p2_encoding=2, p1_turn=p1_turn, size=self.size, last_move=None)

        nn_input = torch.tensor(game.get_nn_input(
            self.net.num_input_channels)).float().unsqueeze(0)
        logits = self.net.forward(nn_input).detach().squeeze(0).cpu().numpy()
        logits = game.mask_illegal_indexes(logits)
        indexes = np.unravel_index(np.argmax(logits), (self.size, self.size))
        return int(indexes[0]), int(indexes[1])
