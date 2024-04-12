import os
import time
import numpy as np
from tqdm import tqdm
import torch
import matplotlib.pyplot as plt

from project2.globals import (
    BOARD_SIZE,
    EXPERIMENT_NAME,
    TOPP_EPSILON,
    TOPP_NUM_GAMES_BETWEEN_ANY_TWO_PLAYERS,
    TOPP_VISUALIZE,
)
from project2.neural_network import ConvNetwork, FeedForwardNetwork, NeuralNetwork
from project2.game import HexGame
import datetime

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

if TOPP_VISUALIZE:
    fig, ax = plt.subplots(1)
    plt.ion()
    fig.show()
    fig.canvas.draw()

ANETS = {}
ANETS["untrained"] = FeedForwardNetwork()
ANETS["supervised"] = ConvNetwork()
ANETS["supervised"].load_state_dict(torch.load("saved_networks/supervised_10.pt", map_location=device))
folder_path = "saved_networks"
for filename in sorted(
    os.listdir(folder_path), key=lambda x: int(x.split("_")[-1].split(".")[0])
):
    # Check if the file is a network file
    if filename.startswith(EXPERIMENT_NAME) and filename.endswith(".pt"):
        # Load the network
        network_path = os.path.join(folder_path, filename)
        network = FeedForwardNetwork()
        network.load_state_dict(torch.load(network_path, map_location=device))
        ANETS[filename.replace(EXPERIMENT_NAME + "_", "")] = network


num_nets = len(ANETS)
wins = {}
for key, value in ANETS.items():
    wins[key] = 0

rounds = num_nets * (num_nets - 1)


def play_a_game(ANET1: NeuralNetwork, ANET2: NeuralNetwork, visualize=False):
    game = HexGame(size=BOARD_SIZE, last_move=None)

    while not game.is_terminal():
        if np.random.rand() < TOPP_EPSILON:
            legal_moves = game.get_legal_moves()
            move = legal_moves[np.random.choice(len(legal_moves))]
            game = game.make_move(move)
            continue

        if game.p1_turn:
            nn_input = torch.tensor(game.get_nn_input(ANET1.num_input_channels)).float().unsqueeze(0)
            logits = ANET1(nn_input).detach().squeeze(0).cpu().numpy()
        else:
            nn_input = torch.tensor(game.get_nn_input(ANET2.num_input_channels)).float().unsqueeze(0)
            logits = ANET2(nn_input).detach().squeeze(0).cpu().numpy()
        assert logits.shape == (game.size, game.size)
        logits = game.mask_illegal_indexes(logits)
        if logits.sum() == 0:
            logits = game.mask_illegal_indexes(np.ones_like(logits))
            print("All legal moves were masked, choosing randomly.")

        move = np.argmax(logits)
        game = game.make_move(move)
        if visualize:
            # Set windows title
            print(f"{ANET1.__class__.__name__} vs {ANET2.__class__.__name__}", end="\r")
            plt.title(f"{ANET1.__class__.__name__} vs {ANET2.__class__.__name__}")
            game.visualize_board(ax)
            time.sleep(0.25)

    return game.get_winner(), game.p1_encoding, game.p2_encoding


def play_a_round(ANET1, ANET2, G):
    wins_p1 = 0
    wins_p2 = 0
    for i in range(G):
        winner, p1_encoding, p2_encoding = play_a_game(
            ANET1, ANET2, visualize=(TOPP_VISUALIZE and i == 0)
        )
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
                    ANET1, ANET2, TOPP_NUM_GAMES_BETWEEN_ANY_TWO_PLAYERS // 2
                )
                wins[filename1] += wins_p1
                wins[filename2] += wins_p2
                pbar.update(1)

# Reset plot
plt.close()
# Plot the results
plt.bar(wins.keys(), wins.values())
plt.xticks(rotation=45)
plt.gcf().subplots_adjust(bottom=0.25)
plt.ylabel("Wins")
plt.title(f"TOPP Results - {EXPERIMENT_NAME}")
timestamp = datetime.datetime.now().strftime("%Y%m%d%H%M%S")
plt.savefig(f"project2/figs/{EXPERIMENT_NAME}_topp_results_{timestamp}.png")
plt.show()
