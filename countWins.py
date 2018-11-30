fin = open("results.data",'r')
p1wins,p2wins,draw = 0,0,0
lastVal = 2.0

for ln in fin:
    lst = ln.strip().split(',')
    val = float(lst[-1])
    if val!=lastVal:
        if val<0:
            p1wins+=1
        elif val>0:
            p2wins+=1
        else:
            draw+=1
        lastVal=val

print("p1:",p1wins,"p2:",p2wins,"draws:",draw)