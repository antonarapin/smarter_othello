class OthelloAgent:
    def __init__(self, problem, d=""):
        self.problem = problem
        #self.weights = [[4,-3,2,2],[-3,-4,-1,-1],[2,-1,1,0],[2,-1,0,1]]
        self.weights = [[99,-8,8,6],[-8,-24,-4,-3],[8,-4,7,4],[6,-3,4,0]]
        if d=="":
            self.depth = 3
        else:
            self.depth = int(d)
        self.exploitCheck = False
        self.cornerList = [(0,0),(7,0),(0,7),(7,7)]
        self.maxVal = 400

    def getMoves(self):
        state = self.problem.getState()
        turn = self.problem.getTurn()
        self.exploitCheck=True
        moveList,mm = self.MinimaxSearch(state,turn,-float('inf'),float('inf'),self.depth)
        return moveList
    
    def MinimaxSearch(self,state,turn,a,b,d):
        if d<=0:
            return [], self.weightedHeuristic(state,turn)
        
        succList = self.problem.getSuccessors(state,turn)
        if self.exploitCheck:
            self.exploitCheck=False
            ex = self.exploits(succList,turn)
            if len(ex)>0:
                return ex,0

        mmList = [None]*len(succList)
        curMMScore = float('inf')
        if turn == 1:
            curMMScore = -float('inf')
        curMMIdx = None
        prune = False


        for i in range(len(succList)):
            if succList[i][3]!=None: #base case, terminal state
                mmList[i] = succList[i][3]/64
            else: 
                toss,mmList[i] = self.MinimaxSearch(succList[i][0],succList[i][2],a,b,d-1)
            if turn==1:
                if mmList[i]>curMMScore:
                    curMMScore = mmList[i]
                    curMMIdx = i
                if (curMMScore>=b):
                    prune = True
                    break
                a = max(a,curMMScore)
            elif turn==-1:
                if mmList[i]<curMMScore:
                    curMMScore = mmList[i]
                    curMMIdx = i
                if (curMMScore<=a):
                    prune = True
                    break
                b = min(b,curMMScore)
        
        return [succList[curMMIdx][1]],curMMScore

    
    def weightedHeuristic(self,state,turn):
        score = [0,0]
        cScore = list(map(lambda x: state[x[0]][x[1]],self.cornerList))
        if sum(cScore)!=0:
            return sum(cScore)/4
        
        for  r in range(8):
            if r<4:
                im = r
            else:
                im = 3 - r%4
            for c in range(8):
                if c<4:
                    ij = c
                else:
                    ij = 3 - c%4
                if state[r][c]!=0:
                    score[(1+state[r][c])//2] += self.weights[im][ij]
        val = (score[1]-score[0]) #so results are between -1 and 1
        self.maxVal = max(self.maxVal,abs(val))
        return val/self.maxVal

    def exploits(self,succList,turn):
        corner,block = [],[]
        for i in range(len(succList)):
            if succList[i][3]!=None:
                if (succList[i][3]*turn) > 0: #if you can win right now, do it
                    return [succList[i][1]]
            if succList[i][1] in self.cornerList:
                corner.append(succList[i][1])
            if succList[i][2]==turn:
                block.append(succList[i][1])
        if len(corner)>0:
            return [corner[0]]
        if len(block)>0:
            return [block[0]]
        return []