import cImage,copy

class Othello:
    def __init__(self, row = 8, col=8):
        self.rows = row
        self.cols = col
        self.turn = -1
        self.board = [[0 for i in range(self.cols)] for j in range(self.rows)]
        self.truState = True

        #initialize the board to have the 4 starting disks
        for r in range(self.rows//2-1,self.rows//2 +1):
            for c in range(self.cols//2-1,self.cols//2 +1):
                if int((r%2 + c%2)%2) ==0:
                    self.board[r][c] = -1
                else:
                    self.board[r][c] = 1
        self.directions = [(x,y) for x in range(-1,2) for y in range(-1,2) if not (x==0 and y==0)]
        self.emptyContours = [] #stores all the empty spaces arround the disks
        for i in range(self.cols//2-2,self.cols//2+2):
            for j in range(self.rows//2-2,self.rows//2+2):
                if not (i in [self.cols//2-1,self.cols//2] and j in [self.rows//2-1,self.rows//2]):
                    self.emptyContours.append((j,i))

    def getState(self):
        '''Returns the game state (the board) as a row major list of lists '''
        return copy.deepcopy(self.board)

    def __setState(self,state,p):
        """takes a row major list of lists and sets the board state"""
        if p not in [1,-1]:
            raise ValueError("Not a valid turn given")
        self.turn = p

        if len(state)!=self.rows:
            raise ValueError("Board is wrong size: " + str(len(state)) + " rows, but expected "+str(self.rows))
        
        self.board = copy.deepcopy(state)
        newContours = []

        for r in range(self.rows):
            if len(self.board[r])!=self.cols:
                raise ValueError("Board is wrong size: " + str(len(self.board[r])) + " cols, but expected "+str(self.cols))
            for c in range(self.cols):
                if self.board[r][c]!=0:
                    for d in self.directions:
                        if not self.__outBounds(r+d[0],c+d[1]):
                            if self.board[r+d[0]][c+d[1]]==0:
                                if not (r+d[0],c+d[1]) in newContours:
                                    newContours.append((r+d[0],c+d[1]))

        self.emptyContours = newContours

        
    def reset(self):
        """Resets the board and gameplay to original config. Returns True on success"""
        self.__init__(self.rows,self.cols)
        return True

    def getSuccessors(self,state,p):
        '''Takes a state and returns the possible successors as 4-tuples: 
        (next state, action to get there, whose turn in the next state, final score).
        For the last two items, see getTurn() and finalScore().'''
        currentState = self.board
        currTurn = self.turn
        succ = []
        self.truState = False
        self.__setState(state,p)
        for a in self.legalMovesOnly():
            self.move(a[0],a[1])
            succ.append((self.getState(), a, self.getTurn(), self.finalScore()))
            self.__setState(state,p)
        self.__setState(currentState,currTurn)
        self.truState = True
        return succ

    def getLegalMoves(self, p=None):
        '''Returns the set of legal moves in the current state.'''
        if p==None:
            p=self.turn
        return [(x[1],x[0][1]) for x in map(lambda x: (self.check_move(x[0],x[1],p),x), self.emptyContours) if x[0][0]]

    def legalMovesOnly(self):
        return [x[1] for x in map(lambda x: (self.check_move(x[0],x[1],self.turn),x), self.emptyContours) if x[0][0]]

    def check_move(self,r,c,p=None):
        if p==None:
            p=self.turn
        if self.board[r][c]!=0:
            return False,[]
        flip_l = []
        for d in self.directions:
            row,col = r+d[0],c+d[1]
            trace = True
            f = []
            while trace:
                if self.__outBounds(row,col) or self.board[row][col]==0:
                    trace = False 
                elif self.board[row][col]==p:
                    if len(f)>0:
                        flip_l+=f
                    trace = False
                else:
                    f.append((row,col))
                row += d[0]
                col += d[1]
        return len(flip_l)!=0, flip_l

    def move(self,r,c,p=None):
        if p==None:
            p=self.turn
        if self.turn == 2:
            raise ValueError("Illegal move: game is terminated.")
        legal,flips = self.check_move(r,c,p)
        if legal:
            for f in flips:
                self.board[f[0]][f[1]] = p
            self.board[r][c] = p

            #update the contours
            self.emptyContours.remove((r,c))
            for d in self.directions:
                if not self.__outBounds(r+d[0],c+d[1]):
                    if self.board[r+d[0]][c+d[1]]==0:
                        if not (r+d[0],c+d[1]) in self.emptyContours:
                            self.emptyContours.append((r+d[0],c+d[1]))

            #determine who's turn it it next
            if self.isTerminal():
                self.turn = 2
            elif self.turn==-1:
                if len(self.getLegalMoves(1))!= 0:
                    self.turn = 1
                elif self.truState:
                    print("Player 2 has no legal moves, therefore they forefeit their turn")
            else:
                if len(self.getLegalMoves(-1))!= 0:
                    self.turn = -1
                elif self.truState:
                    print("Player 1 has no legal moves, therefore they forefeit their turn")
        else:
            print("Illegal move")
    

    def __outBounds(self,r,c):
        return r<0 or c<0 or r>=self.rows or c>=self.cols

    def isTerminal(self):
        if len(self.getLegalMoves(1))== 0 and len(self.getLegalMoves(-1))== 0:
            return True
        for l in self.board:
            if 0 in l:
                return False
        return True

    def finalScore(self):
        '''If the game is not over, returns None.
        If it is over, it returns the net number of discs the winner received.
        If nevative, player 1 won. If positive, player 2 won. If zero, draw.'''
        if self.isTerminal():
            counter = 0
            for row in self.board:
                counter+=sum(row)
            return counter
        return None
    
    def getTurn(self):
        return self.turn
    
    def getTile(self,r,c):
        return self.board[r][c]
    
import time
class OthelloDisplay:
    '''Displays a Connect Four game.'''
    def __init__(self, board):
        '''Takes a ConnectFour and initializes the display.'''
        self.__board = board
        
        self.__numCols = 8
        self.__numRows = 8

        self.__images = []
        for r in range(self.__numRows):
            self.__images.append([])
            for c in range(self.__numCols):
                self.__images[r].append([])
                self.__images[r][c].append(cImage.FileImage("blank.gif"))
                self.__images[r][c].append(cImage.FileImage("p1disc.gif"))
                self.__images[r][c].append(cImage.FileImage("p2disc.gif"))
                for i in range(3):
                    img = self.__images[r][c][i]
                    img.setPosition(c*img.getWidth(), r*img.getHeight())

        self.__tileWidth = self.__images[0][0][0].getWidth()
        self.__tileHeight = self.__images[0][0][0].getHeight()
        self.__win = cImage.ImageWin("Othello/Reversi!", self.__numCols*self.__tileWidth, self.__numRows*self.__tileHeight)
        self.update()

    def update(self):
        '''Updates the game display based on the game's current state.'''
        for r in range(self.__numRows):
            for c in range(self.__numCols):
                t = self.__board.getTile(r, c)
                if t == 0:
                    self.__images[r][c][0].draw(self.__win)
                elif t==-1:
                    self.__images[r][c][1].draw(self.__win)
                else:
                    self.__images[r][c][2].draw(self.__win)

        time.sleep(0.1)
                            
    def getMove(self):
        '''Allows the user to click to decide which column to move in.'''
        pos = self.__win.getMouse()
        col = pos[0]//self.__tileWidth
        row = pos[1]//self.__tileHeight
        while not self.__board.check_move(row,col):
            print("Illegal move! Please click on an empty space that would complete a proper move")
            pos = self.__win.getMouse()
            col = pos[0]//self.__tileWidth
            row = pos[1]//self.__tileHeight
        return (row,col)

    def exitonclick(self):
        self.__win.exitonclick()