import numpy as np

#Choose a range of x grid points (dx in cm1)
dxmin = 20
dxmax = 30
#Choose a range of y grid points (dy in cm1)
dymin =20
dymax = 30

#Choose the number of 36-processor nodes to use (select in cm1)
selectmin = 1
selectmax = 10

#This is the number of xnodes (nodex in cm1). Dont change the values here
nodexmin = 1#1int(np.sqrt(selectmin*36)) - 1
nodexmax = 50#int(np.sqrt(selectmax*36)) + 1

#This is the number of ynodes (nodey in cm1). Dont change the values here
nodeymin = 1#int(np.sqrt(selectmin*36)) - 1
nodeymax = 50#int(np.sqrt(selectmax*36)) + 1



for dx in range(dxmin,(dxmax+1)):
    for dy in range(dymin,(dymax+1)):
        for select in range(selectmin,(selectmax+1)):
            for nodex in range(nodexmin,(nodexmax+1)):
                for nodey in range(nodeymin,(nodeymax+1)):
                    if nodex*nodey == select*36 and dx%nodex == 0 and dy%nodey == 0 and dx/nodex >= 3 and dy/nodey >=3:
                        print ('dx=',dx,'dy=',dy,'select=',select,'nodex=',nodex,'nodey=',nodey)
                        print (select)
