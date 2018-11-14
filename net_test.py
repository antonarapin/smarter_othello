import random 
from ffdnn import simple_net

net = simple_net(num_hid=4,num_in=3,num_out=3,num_l=2,lr=0.08)

num_exp = 80000
for i in range(num_exp):
    x, y, z = random.randint(0,1), random.randint(0,1), random.randint(0,1)
    a = int(x and y)
    b = int(y and z)
    c = int(x and z)
    net.train([(x,y,z)],[(a,b,c)])

for k in range(10):
    x, y, z = random.randint(0,1), random.randint(0,1), random.randint(0,1)
    out = net.predict([(x,y,z)])
    print("should be:   ",((x and y),(y and z),(x and z)))
    print("res:   ",out)
