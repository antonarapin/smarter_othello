import random
import argparse
import time
import importlib
import signal
from othello import *

class timeout:
    def __init__(self, seconds=1):
        self.seconds = seconds

    def timeoutCallback(self, signum, frame):
        raise TimeoutError("Timed out after " + str(self.seconds) + " seconds")

    def __enter__(self):
        signal.signal(signal.SIGALRM, self.timeoutCallback)
        signal.alarm(self.seconds)

    def __exit__(self, type, value, traceback):
        signal.alarm(0)    

def main():
    parser = argparse.ArgumentParser(description='Play Othello with computer or human players.')
    parser.add_argument('-p1', '--player1', type=str, default='random', help='the name of a Python file containing an Othello agent, or "random" or "human" (default: random)')
    parser.add_argument('-p2', '--player2', type=str, default='random', help='the name of a Python file containing a Othello agent, or "random" or "human" (default: random)')
    parser.add_argument('-t', '--trials', type=int, default=1, help='the number of games to play (default 1; has no effect if either player is human; if TRIALS > 1 game will not be displayed)')
    parser.add_argument('-nd', '--nodisplay', action='store_true', default=False, help='do not display the game (has no effect if a player is human)')
    
    args = parser.parse_args()

    problem = Othello()   

    players = [args.player1, args.player2]
        
    if args.player1 == "human" or args.player2 == "human":
        args.trials = 1
        args.nodisplay = False
        print("Please click on a column with an empty space.")       

    if args.trials == 1 and not args.nodisplay:
        display = OthelloDisplay(problem)
        
    playerPrograms = [None, None]
    for i in range(2):
        if players[i] != "random" and players[i] != "human":
            mod = importlib.import_module(".".join(players[i].split(".")[:-1]))
            with timeout(2):
                playerPrograms[i] = mod.OthelloAgent(problem)
              
    p1Wins = 0
    p2Wins = 0
    draws = 0
    times = [0, 0]
    turns = [0, 0]
    for t in range(args.trials):
        problem.reset()
       
        while not problem.isTerminal():
            turn = problem.getTurn()
            playerIdx = (1 + turn)//2
            if players[playerIdx] == "random":
                moves = problem.legalMovesOnly()
            elif players[playerIdx] == "human":
                moves = [display.getMove()]
            else: #minimax
                startT = time.time()
                try:
                    with timeout(3):
                        moves = playerPrograms[playerIdx].getMoves()
                except TimeoutError:
                    print("Player " + str(playerIdx+1) + " timed out after 3 seconds. Choosing random action.")
                    moves = problem.legalMovesOnly()
                endT = time.time()
                times[playerIdx] += endT - startT
                turns[playerIdx] += 1

            move = random.choice(moves)
            problem.move(move[0],move[1])                
            if args.trials == 1 and not args.nodisplay:
                display.update()
    
        FS = problem.finalScore()
        if FS==0:
            whoWon = "Draw"
            draws += 1
        elif FS > 0:
            whoWon = "Player 2 wins by "+str(FS)+"!"
            p2Wins += 1
        elif FS < 0:
            whoWon = "Player 1 wins by "+str(abs(FS))+"!"
            p1Wins += 1
            
        if args.trials > 1:
            whoWon = "Game " + str(t+1) + ": " + whoWon
        print(whoWon)

    if args.trials > 1:
        print("Stats:")
        print("Player 1 Wins: " + str(p1Wins))
        print("Player 2 Wins: " + str(p2Wins))
        print("Draws:         " + str(draws))
        
    for i in range(2):
        if turns[i] > 0:
            print("Player " + str(i+1) + ": " + str(times[i]/turns[i]) + " seconds per step, on average")

    if args.trials == 1 and not args.nodisplay:
        print("Click on the window to exit")
        display.exitonclick()
            
if __name__ == "__main__":
    main()
