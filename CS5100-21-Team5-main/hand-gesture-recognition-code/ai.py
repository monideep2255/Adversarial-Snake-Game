import multiprocessing

from snakegame import SnakeGame

def playwithAgent(Agent):
    game = SnakeGame(Agent)
    game.startGame()

def playwithHuman():
    game = SnakeGame("human")
    game.startGame()

if __name__ == "__main__":

    agent = "bfs"

    p1 = multiprocessing.Process(target=playwithAgent, args=("bfs", ))
    p2 = multiprocessing.Process(target=playwithHuman, args=())

    p1.start()
    # starting process 2
    p2.start()

    p1.join()
    # wait until process 2 is finished
    p2.join()
    print("Done")