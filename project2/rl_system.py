from project2.game import HexGame
from project2.globals import ANET_M, BOARD_SIZE, NUMBER_OF_EPISODES
from project2.mcts import MCTS
from project2.neural_network import NeuralNetwork

RBUF = []
ANET = NeuralNetwork() 

for i in range(1, NUMBER_OF_EPISODES + 1):
    game = HexGame(BOARD_SIZE)
    mcts = MCTS(ANET=ANET, game_state=game)
    while not game.is_terminal():
        new_game, distribution = mcts.search()
        RBUF.append((game, distribution))
        print("Made actual move")
        game = new_game
    X = [game.get_state() for game, _ in RBUF]
    Y = [distribution for _, distribution in RBUF]
    
    ANET.train_one_batch(X, Y)
    if i % (NUMBER_OF_EPISODES // ANET_M) == 0:
        ANET.save(i)

