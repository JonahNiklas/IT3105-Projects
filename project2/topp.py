import os
import numpy as np
from tqdm import tqdm
import torch
import matplotlib.pyplot as plt

from project2.globals import BOARD_SIZE, EXPERIMENT_NAME, TOPP_EPSILON, TOPP_NUM_GAMES_BETWEEN_ANY_TWO_PLAYERS
from project2.neural_network import NeuralNetwork
from project2.game import HexGame
import datetime

ANETS = {}
folder_path = 'saved_networks'
for filename in os.listdir(folder_path):
    # Check if the file is a network file
    if filename.startswith(EXPERIMENT_NAME) and filename.endswith('.pt'):
        # Load the network
        network_path = os.path.join(folder_path, filename)
        network = NeuralNetwork()
        network.load_state_dict(torch.load(network_path))
        ANETS[filename.replace(EXPERIMENT_NAME + "_", "")] = network

num_nets = len(ANETS)
wins = {}
for (key, value) in ANETS.items():
    wins[key] = 0

rounds = num_nets * (num_nets - 1) // 2


def play_a_game(ANET1, ANET2):
    game = HexGame(size=BOARD_SIZE, last_move=None)

    while not game.is_terminal():
        if np.random.rand() < TOPP_EPSILON:
            legal_moves = game.get_legal_moves()
            move = legal_moves[np.random.choice(len(legal_moves))]
            game = game.make_move(move)
            continue

        nn_input = torch.tensor(game.get_nn_input()).float().unsqueeze(0)
        if game.p1_turn:
            logits = ANET1(nn_input).detach().cpu().numpy()
        else:
            logits = ANET2(nn_input).detach().cpu().numpy()
        logits = game.mask_illegal_indexes(logits)

        move = np.argmax(logits)
        game = game.make_move(move)

    return game.get_winner(), game.p1_encoding, game.p2_encoding


def play_a_round(ANET1, ANET2, G):
    wins_p1 = 0
    wins_p2 = 0
    for _ in range(G):
        winner, p1_encoding, p2_encoding = play_a_game(ANET1, ANET2)
        if winner == p1_encoding:
            wins_p1 += 1
        elif winner == p2_encoding:
            wins_p2 += 1
    return wins_p1, wins_p2


with tqdm(total=rounds, desc="TOPP Progress") as pbar:
    for i, (filename1, ANET1) in enumerate(ANETS.items()):
        for j, (filename2, ANET2) in enumerate(ANETS.items()):
            if i != j:
                wins_p1, wins_p2 = play_a_round(
                    ANET1, ANET2, TOPP_NUM_GAMES_BETWEEN_ANY_TWO_PLAYERS // 2)
                wins[filename1] += wins_p1
                wins[filename2] += wins_p2
                pbar.update(1)

# Plot the results
plt.bar(wins.keys(), wins.values())
plt.xticks(rotation=45)
plt.gcf().subplots_adjust(bottom=0.25)
plt.ylabel('Wins')
plt.title(f'TOPP Results - {EXPERIMENT_NAME}')
timestamp = datetime.datetime.now().strftime("%Y%m%d%H%M%S")
plt.savefig(f'project2/figs/{EXPERIMENT_NAME}_topp_results_{timestamp}.png')
plt.show()

