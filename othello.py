import cImage
import board_sim

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