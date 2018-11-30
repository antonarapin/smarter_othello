import mlp, numpy as np
class OthelloAgent:
    def __init__(self, problem):
        self.problem = problem
        self.net = mlp.mlp(weight1File="fstWeights.data",weight2File="sndWeights.data")

    def getMoves(self):
        state = self.problem.getState()
        turn = self.problem.getTurn()
        successors = self.problem.getSuccessors(state,turn)
        moveList,mm = self.MinimaxSearch(state,turn,-float('inf'),float('inf'),0)

        return moveList
    
    def MinimaxSearch(self,state,turn,a,b,d):
        succList = self.problem.getSuccessors(state,turn)
        curMMScore = float('inf')
        if turn == 1:
            curMMScore = -float('inf')
        curMMIdx = None

        if d<=0:
            preArr = []
            for i in range(len(succList)):
                tmp = []
                for r in succList[i][0]:
                    tmp = tmp+r
                tmp.append(-1)
                preArr.append(tmp)
            arr = np.array(preArr)
            result = self.net.mlpfwd(arr)
            for i in range(len(succList)):
                if turn==1:
                    if result[i]>curMMScore:
                        curMMScore = result[i]
                        curMMIdx = i
                elif turn ==-1:
                    if result[i]<curMMScore:
                        curMMScore = result[i]
                        curMMIdx = i
            return [succList[curMMIdx][1]], curMMScore     
        
        mmList = [None]*len(succList)
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

#after 1st training with 11500 dpts, loses by 16 as p1 to regular greedy. Wins against regular greedy by 12 as p2.
#after 2nd training with 30000 dpts, wins by 16 as p1 against greedy. Wins against greedy by 20 as p2.
#also after 2nd training, when it plays itself, p2 wins by 50...
#after training 3 with 65000 dpts, wins as p1 by 28. Wins as p2 by 40.