class OthelloAgent:
    def __init__(self, problem):
        self.problem = problem

    def getMoves(self):
        state = self.problem.getState()
        turn = self.problem.getTurn()
        successors = self.problem.getSuccessors(state,turn)
        moveList,mm = self.MinimaxSearch(state,turn,-float('inf'),float('inf'),4)

        return moveList
    
    def MinimaxSearch(self,state,turn,a,b,d):
        if d<=0:
            return [], self.greedyHeuristic(state)
        
        succList = self.problem.getSuccessors(state,turn)
        mmList = [None]*len(succList)
        curMMScore = float('inf')
        if turn == 1:
            curMMScore = -float('inf')
        curMMIdx = None
        prune = False

        ###move ordering
        # MO = []
        # for s in range(len(succList)):
        #     if succList[s][0] in self.bank:
        #         MO.append((self.bank[succList[s][0]],s))
        #     else:
        #         MO.append((0,s))
        # MO.sort()
        # if turn==1:
        #     MO.reverse()

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

    
    def greedyHeuristic(sef,state):
        counter = 0
        for row in state:
            counter+=sum(row)
        return counter/64 #div by 64 to ensure value between -1 and 1