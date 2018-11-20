fin = open("results.data",'r')
fout = open("endgameResults.data",'w')

for ln in fin:
    lst = ln.strip().split(',')
    count = 0
    for i in range(len(lst)-1):
        if float(lst[i])!=0:
            count+=1
        
    if count>20:
        fout.write(ln)