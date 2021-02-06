import timeit
import numpy as np
import numpy.linalg as nplin
import matplotlib.pyplot as plt
from matplotlib import cm
from numpy.polynomial.polynomial import polyfit


maxIter = 300 

Re = 220


q  = 9 
c  = np.array([(x,y) for x in [0,-1,1] for y in [0,-1,1]]) 
i1 = np.arange(q)[np.asarray([ci[0]<0  for ci in c])] 
i2 = np.arange(q)[np.asarray([ci[0]==0 for ci in c])] 
i3 = np.arange(q)[np.asarray([ci[0]>0  for ci in c])] 


t = np.array([4/9, 1/9, 1/9, 1/9, 1/36, 1/36, 1/9, 1/36, 1/36])

def equilibrium(rho,u):
    cu   = np.dot(c,u.transpose(1,0,2))
    usqr = u[0]**2+u[1]**2
    feq  = np.zeros((q,nx,ny))
    for i in range(q):
        feq[i,:,:] = rho*t[i] + rho*t[i]*(3*cu[i] + (9/2)*cu[i]**2 - (3/2)*usqr)
    return feq

# # grid size changing for simulation
nx_v = np.asarray([104,260,364,520,676,780,936,1040])
ny_v = np.asarray([36,90,126,180,234,270,324,360])
nxny = np.zeros(nx_v.shape[0])
nxny[:] = nx_v[:] * ny_v[:]
itergrid = len(nxny)

# # Main loop
duration = np.zeros(itergrid)
for it in range(itergrid):
    start = timeit.default_timer()

    nx = nx_v[it]
    ny = ny_v[it]
    print('nx={}, ny={}'.format(nx,ny))
    ly = ny-1

    cx = nx/4
    cy = ny/2
    r  = ny/9

    uLB     = 0.04

    nulb  = uLB*r/Re
    omega = 1/(3*nulb+0.5)


    vel = np.fromfunction(lambda d,x,y: (1-d)*uLB*(1+1e-4*np.sin(y/ly*2*np.pi)),(2,nx,ny))

    obstacle = np.fromfunction(lambda x,y: (x-cx)**2+(y-cy)**2<r**2, (nx,ny))
    
    feq = equilibrium(1.0,vel)
    fin = feq.copy()

    # Loop for time
    for Iter in range(maxIter):
        fin[i1,-1,:] = fin[i1,-2,:]

        rho = sum(fin)
        u   = np.dot(c.transpose(), fin.transpose((1,0,2)))/rho

        u[:,0,:] = vel[:,0,:]

        rho[0,:] = 1/(1-u[0,0,:]) * (sum(fin[i2,0,:])+2*sum(fin[i1,0,:]))

        feq = equilibrium(rho,u)

        fin[i3,0,:] = fin[i1,0,:] + feq[i3,0,:] - fin[i1,0,:]

        fout = fin - omega * (fin - feq)

        noslip = [c.tolist().index((-c[i]).tolist()) for i in range(q)]

        for i in range(q):
            fout[i,obstacle] = fin[noslip[i],obstacle]

        for i in range(q):
            amat = np.roll(fout[i,:,:],c[i,0],axis=0)
            fin[i,:,:] = np.roll(amat,c[i,1],axis=1)
        #----------------------------
        # Timing
        #----------------------------
        if (Iter%100==0):
            print(Iter)
    stop = timeit.default_timer()
    T = stop - start
    duration[it] = T
    print('T = ', T)

#------------------------------------------
# Output : comparing the experiences
#------------------------------------------
print('nxny = ', nxny)
print('duration = ', duration)
b, m = polyfit(nxny, duration, 1)
plt.xlabel('size of system (nx$\cdot$ny)')
plt.ylabel('time (s)')
plt.plot(nxny, duration, '.')
plt.plot(nxny, b + m * nxny, '-')
plt.savefig("serial_sizevtime.png", bbox_inches='tight', dpi=300)
