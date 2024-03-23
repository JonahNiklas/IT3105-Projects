from project2.game import Game, HexGame
from project2.globals import ANET_M, BOARD_SIZE, NUMBER_OF_EPISODES, VISUALIZE
from project2.mcts import MCTS
from project2.neural_network import NeuralNetwork
import numpy as np
import matplotlib.pyplot as plt

if VISUALIZE:
    fig, ax = plt.subplots(1)
    plt.ion()
    fig.show()
    fig.canvas.draw()


RBUF = []
ANET = NeuralNetwork() 

for i in range(1, NUMBER_OF_EPISODES + 1):
    game = HexGame(size=BOARD_SIZE, last_move=None)
    mcts = MCTS(ANET=ANET, game_state=game)
    move_num = 0
    while not game.is_terminal():
        new_game, distribution = mcts.search()
        RBUF.append((game, distribution))
        move_num += 1
        print(f"Made actual move {move_num} of {BOARD_SIZE**2} possible", end="\r")
        game = new_game
        if VISUALIZE: game.visualize_board(ax)
    if VISUALIZE: print(f"Player {game.get_winner()} won the game")
    X = np.array([game.get_state() for game, _ in RBUF])
    Y = np.array([distribution for _, distribution in RBUF])
    
    ANET.train_one_batch(X, Y)
    if i % (NUMBER_OF_EPISODES // ANET_M) == 0:
        ANET.save(i)

