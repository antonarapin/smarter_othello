class othello_board:

    def __init__(self,nrows,ncols):
        self.rows = nrows
        self.cols = ncols
        self.turn = 0
        self.board = [[None for i in range(self.cols)] for j in range(self.rows)]

        #initialize the board to have the 4 starting disks
        for r in range(nrows//2-1,nrows//2 +1):
            for c in range(ncols//2-1,ncols//2 +1):
                self.board[r][c] = int((r%2 + c%2)%2)
        self.directions = [(x,y) for x in range(-1,2) for y in range(-1,2) if not (x==0 and y==0)]
        self.empty_spaces = []
        for i in range(ncols//2-2,ncols//2+2):
            for j in range(nrows//2-2,nrows//2+2):
                if not (i in [ncols//2-1,ncols//2] and j in [nrows//2-1,nrows//2]):
                    self.empty_spaces.append((j,i))

    def _legal_check(func):
        def wrapper(self, *args, **kwargs):
            return False if not self.check_move(args[0],args[1],args[2])[0] else func(self, *args, **kwargs)  
        return wrapper

    def check_move(self,r,c,p):
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

    @_legal_check
    def make_move(self,r,c,p,flips):
        for k in flips:
            self.board[k[0]][k[1]]=p
        self.board[r][c] = p
        self.empty_spaces.remove((r,c))
        for d in self.directions:
            if not self.is_out(r+d[0],c+d[1]):
                if self.board[r+d[0]][c+d[1]]==None:
                    if not (r+d[0],c+d[1]) in self.empty_spaces:
                        self.empty_spaces.append((r+d[0],c+d[1]))
        return True

    def is_out(self,r,c):
        return r<0 or c<0 or r>=self.rows or c>=self.cols

    def get_num_disks(self,p):
        s = 0
        for r in range(self.rows):
            s+=sum([1 for x in self.board[r] if x==p])
        return s

    def get_legal_moves(self,p):
        return [(x[1],x[0][1]) for x in map(lambda x: (self.check_move(x[0],x[1],p),x), self.empty_spaces) if x[0][0]]


#short test of the functions
if __name__ == "__main__":
    b = othello_board(8,8) 
    lm = b.get_legal_moves(1)
    for r in b.board:
        print(r)
    print(lm)
    print(b.legalMovesOnly(1))
    b.make_move(lm[0][0][0],lm[0][0][1],1,lm[0][1])
    for r in b.board:
        print(r)
    print(b.empty_spaces)
