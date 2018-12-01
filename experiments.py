import sys,os

print("\nWeighted Discs Round Robin test\n_______________________________\n")
for i in range(1,4):
    for j in range(i,4):
        print("\nNow Trying p1 at d=",i,"and p2 at d=",j)
        os.system("python3 playOthello.py -nd -p1 othelloAgent2.py -p2 othelloAgent3.py -p1A "+str(i)+" -p2A "+str(j))
        print("\nNow Trying p1 at d=",j,"and p2 at d=",i)
        os.system("python3 playOthello.py -nd -p1 othelloAgent2.py -p2 othelloAgent3.py -p1A "+str(j)+" -p2A "+str(i))