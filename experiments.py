import sys,os

print("\nWeighted Discs Round Robin test\n_______________________________\n")
#fst = "mlpLearnedWeights/endfWt"
#snd = "mlpLearnedWeights/endsWt"
for i in range(3,4):
    for j in range(1,6):
        # print("\nNow Trying agent1 as p1 at d=",i,"and mlp as p2 at pts=",j)
        # os.system("python3 playOthello.py -nd -p1 othelloAgent1.py -p2 othelloNetAgent1.py -p1A "+str(i)+" -p2A "+fst+str(j)+".data%"+snd+str(j)+".data")
        # print("\nNow Trying mlp as p1 at d=",i,"and agent1 as p2 at pts=",j)
        # os.system("python3 playOthello.py -nd -p1 othelloNetAgent1.py -p2 othelloAgent1.py -p2A "+str(i)+" -p1A "+fst+str(j)+".data%"+snd+str(j)+".data")

        print("\nNow Trying mlp as p1 at d=",i,"and agent2 as p2 at pts=",j)
        os.system("python3 playOthello.py -nd -p1 othelloNetAgent1.py -p2 othelloAgent2.py -p2A "+str(i)+" -p1A "+"fWt0."+str(j)+"RR.data%"+"sWt0."+str(j)+"RR.data")
        print("\nNow Trying agent2 as p1 at d=",i,"and mlp as p2 at pts=",j)
        os.system("python3 playOthello.py -nd -p1 othelloAgent2.py -p2 othelloNetAgent1.py -p1A "+str(i)+" -p2A "+"fWt0."+str(j)+"RR.data%"+"sWt0."+str(j)+"RR.data")
