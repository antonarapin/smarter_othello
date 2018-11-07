import cImage

class Othello:
    def __init__(self, row = 8, col=8):
        self.rows = row
        self.cols = col
        self.turn = 0
        self.board = [[None for i in range(self.cols)] for j in range(self.rows)]

        #initialize the board to have the 4 starting disks
        for r in range(nrows//2-1,nrows//2 +1):
            for c in range(ncols//2-1,ncols//2 +1):
                self.board[r][c] = int((r%2 + c%2)%2)
        self.directions = [(x,y) for x in range(-1,2) for y in range(-1,2) if not (x==0 and y==0)]
        self.emptyContours = []
        for i in range(ncols//2-2,ncols//2+2):
            for j in range(nrows//2-2,nrows//2+2):
                if not (i in [ncols//2-1,ncols//2] and j in [nrows//2-1,nrows//2]):
                    self.emptyContours.append((j,i))

    def getState(self):
        '''Returns the game state (the board) as a row major list of lists '''
        return self.board

    def setState(self):
        """I hope I'll never need this. 
        If I need to evaluate something about a game state,
        pass in the state you want to check"""
        pass

    def getLegalMoves(self, p=self.turn):
        '''Returns the set of legal moves in the current state (a move is a column index).'''
        return [(x[1],x[0][1]) for x in map(lambda x: (self.check_move(x[0],x[1],p),x), self.empty_spaces) if x[0][0]]

    def check_move(self,r,c,p=self.turn):
        if self.board[r][c]!=None:
            return False,[]
        flip_l = []
        for d in self.directions:
            row,col = r+d[0],c+d[1]
            trace = True
            f = []
            while trace:
                if self.is_out(row,col) or self.board[row][col]==None:
                    trace = False 
                elif self.board[row][col]==p:
                    flip_l+=f
                    trace = False
                else:
                    f.append((row,col))
                row += d[0]
                col += d[1]
        return len(flip_l)!=0, flip_l

    def move(self,r,c,p=self.turn):
        pass
    
    def isTerminal(self):
        for l in self.board:
            if None in l:
                return False
        return True
    
    def getTurn(self):
        return self.turn
    
    def getTile(self,r,c):
        return self.board[r][c]
    

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
                self.__images[r][c].append(cImage.FileImage("c4Blank.gif"))
                self.__images[r][c].append(cImage.FileImage("c4Max.gif"))
                self.__images[r][c].append(cImage.FileImage("c4Min.gif"))
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
                if t == None:
                    self.__images[r][c][0].draw(self.__win)
                elif t == 1:
                    self.__images[r][c][1].draw(self.__win)
                else: #"O"
                    self.__images[r][c][2].draw(self.__win)
                            
    def getMove(self):
        '''Allows the user to click to decide which column to move in.'''
        pos = self.__win.getMouse()
        col = pos[0]//self.__tileWidth
        row = pos[1]//self.__tileHeight
        while not self.__board.check_move(row,col,self.__board.getTurn()):
            print("Illegal move! Please click on an empty space that would complete a proper move")
            pos = self.__win.getMouse()
            col = pos[0]//self.__tileWidth
            row = pos[1]//self.__tileHeight
        return (row,col)

    def exitonclick(self):
        self.__win.exitonclick()