from matplotlib import pyplot as plt
from project2.game import HexGame
from project2.hex_actor import HexActor


bot = HexActor(
    network_filepath="./saved_networks/conv7x7-500-sims_anet_28.pt", size=7)

board = HexGame(7, last_move=None)

fig, ax = plt.subplots(1)
plt.ion()
fig.show()
fig.canvas.draw()

while True:
    move = input("Enter your move (row, col): ")
    row, col = map(int, move.split(","))
    board = board.make_move((row, col))
    print(board)
    if board.is_terminal():
        print("You win!")
        break
    b = [1] + board.get_state().flatten().tolist()
    bot_move = bot.get_action(b)
    print(f"Bot moves: {bot_move}")
    board = board.make_move(bot_move)
    if board.is_terminal():
        print("Bot wins!")
        break

    board.visualize_board(ax)
