for e in range(63,65,10):
    fin = open("results.data",'r')
    fout = open("end"+str(e)+".data",'w')
    for ln in fin:
        lst = ln.strip().split(',')
        count = 0
        for i in range(len(lst)-1):
            if float(lst[i])!=0:
                count+=1
            
        if count>e:
            fout.write(ln)
    fin.close()
    fout.close()