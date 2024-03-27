import os
import time

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.multiprocessing as mp
from tqdm import tqdm

from project2.game import Game, HexGame
from project2.globals import ANET_M, BOARD_SIZE, NUMBER_OF_EPISODES, VISUALIZE
from project2.mcts import MCTS
from project2.neural_network import NeuralNetwork

if VISUALIZE:
    fig, ax = plt.subplots(1)
    plt.ion()
    fig.show()
    fig.canvas.draw()


def play_game(process_id, ANET, RBUF, RBUF_lock, game_num, game_num_lock):
    torch.set_num_threads(1)
    while game_num.value < NUMBER_OF_EPISODES:
        game = HexGame(size=BOARD_SIZE, last_move=None)
        mcts = MCTS(ANET=ANET, game_state=game)
        move_num = 0
        local_RBUF = []
        while not game.is_terminal():
            new_game, distribution = mcts.search()
            local_RBUF.append((game, distribution))
            move_num += 1
            # print(
            #     f"Process {process_id}\tMade actual move {move_num} of {BOARD_SIZE**2} possible",
            #     end="\r",
            # )
            game = new_game
            if VISUALIZE:
                game.visualize_board(ax)
        with game_num_lock:
            game_num.value += 1
            if game_num.value % (NUMBER_OF_EPISODES // ANET_M) == 0:
                ANET.save(i)
        if VISUALIZE:
            print(f"Process {process_id}\tGame {game_num.value}: Player {game.get_winner()} won the game")

        with RBUF_lock:
            RBUF.extend(local_RBUF)
        X = np.array([game.get_nn_input() for game, _ in RBUF])
        Y = np.array([distribution for _, distribution in RBUF])

        ANET.train_one_batch(X, Y)


if __name__ == "__main__":
    mp.set_start_method("spawn")
    ANET = NeuralNetwork()
    ANET.share_memory()
    RBUF = mp.Manager().list()
    RBUF_lock = mp.Lock()
    game_num = mp.Value("i", 0)
    game_num_lock = mp.Lock()
    num_processes = os.cpu_count()
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
