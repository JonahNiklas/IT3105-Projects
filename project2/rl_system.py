import os
import time

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.multiprocessing as mp
from tqdm import tqdm

from project2.game import Game, HexGame
from project2.globals import *
from project2.mcts import MCTS
from project2.neural_network import ConvNetwork, NeuralNetwork

if VISUALIZE:
    fig, ax = plt.subplots(1)
    plt.ion()
    fig.show()
    fig.canvas.draw()

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def play_game(process_id, ANET: NeuralNetwork, RBUF, RBUF_lock, game_num, game_num_lock):
    torch.set_num_threads(1)
    while game_num.value < NUMBER_OF_EPISODES:
        game = HexGame(size=BOARD_SIZE, last_move=None)
        mcts = MCTS(ANET=ANET, game_state=game)
        local_RBUF = []
        while not game.is_terminal():
            new_game, distribution = mcts.search()
            local_RBUF.append(
                (
                    torch.tensor(game.get_nn_input(ANET.num_input_channels), dtype=torch.float32).to(device),
                    torch.tensor(distribution, dtype=torch.float32).to(device),
                )
            )
            game = new_game
            if VISUALIZE:
                game.visualize_board(ax)
        with game_num_lock:
            game_num.value += 1
            if game_num.value % (NUMBER_OF_EPISODES // ANET_M) == 0:
                print(f"Saving net at game number {game_num.value}")
                ANET.save(game_num.value)
        if VISUALIZE:
            print(
                f"Process {process_id}\tGame {game_num.value}: Player {game.get_winner()} won the game"
            )
        with RBUF_lock:
            RBUF.extend(local_RBUF)
            RBUF = RBUF[-MAX_RBUF_SIZE:]

        ANET.train_one_batch(RBUF)

    print(f"Process {process_id} done!")


if __name__ == "__main__":
    mp.set_start_method("spawn")
    ANET = ConvNetwork()
    if START_RL_FROM_STATE is not None:
        print(f"Starting RL from state {START_RL_FROM_STATE}")
        ANET.load_state_dict(torch.load(f"saved_networks/{START_RL_FROM_STATE}", map_location=device))
    ANET.to(device)
    if not torch.cuda.is_available():
        ANET.share_memory() 
    RBUF = mp.Manager().list()
    RBUF_lock = mp.Lock()
    game_num = mp.Value("i", 0)
    game_num_lock = mp.Lock()
    num_processes = os.cpu_count()
    print(f"Using {num_processes} processes for training ANET")
    processes = []
    for i in range(num_processes):
        p = mp.Process(
            target=play_game, args=(i, ANET, RBUF, RBUF_lock, game_num, game_num_lock)
        )
        p.start()
        processes.append(p)
    last_completed_game = 0
    with tqdm(total=NUMBER_OF_EPISODES, desc="Total Training Progress") as pbar:
        while game_num.value < NUMBER_OF_EPISODES:
            pbar.update(game_num.value - last_completed_game)
            last_completed_game = game_num.value
            time.sleep(10)

    for p in processes:
        p.join()
