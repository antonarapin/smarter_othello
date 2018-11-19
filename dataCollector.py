class OthelloDataCollector:
    def __init__(self,writeFile = None):
        """If no file is given, data is written to a new file. 
        If a file is given, data is appended."""
        if writeFile==None:
            self.__fname = "results.data"
        else:
            self.__fname = writeFile
        self.__dataPointsAdded = 0
        self.__data = []

    def addDataPoint(self, board):
        """Takes a list of lists representing the current state of the board.
        Stores the datapoint for future write to file."""
        tmp = []
        for r in board:
            tmp = tmp+r
        self.__data.append(tmp)
        self.__dataPointsAdded+=1
    
    def endGame(self, result):
        """Call when the game is over and data should be written to the file.
        Takes the net result of the game."""
        fin = open(self.__fname,"a")
        result = result/64
        
        for pt in self.__data:
            pt.append(result)
            ln = ",".join(str(i) for i in pt)+"\n"
            fin.write(ln)
        
        fin.close()
        self.__data = []
        return self.__dataPointsAdded



