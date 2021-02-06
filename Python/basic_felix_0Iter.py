#!/usr/bin/env python
# coding: utf-8


import timeit
import numpy as np
import numpy.linalg as nplin
import matplotlib.pyplot as plt
from matplotlib import cm
from numpy.polynomial.polynomial import polyfit



maxIter = 20000 

Re      = 220

nx = 520
ny = 180 
ly = ny-1 
q  = 9

cx = nx/4 
cy = ny/2 
r  = ny/9


uLB     = 0.04

nulb  = uLB*r/Re; 
omega = 1/(3*nulb+0.5)


vel = np.fromfunction(lambda d,x,y: (1-d)*uLB*(1+1e-4*np.sin(y/ly*2*np.pi)),(2,nx,ny))

obstacle = np.fromfunction(lambda x,y: (x-cx)**2+(y-cy)**2<r**2, (nx,ny))

c = np.array([(x,y) for x in [0,-1,1] for y in [0,-1,1]]) 

i1 = np.arange(q)[np.asarray([ci[0]<0  for ci in c])] 
i2 = np.arange(q)[np.asarray([ci[0]==0 for ci in c])] 
i3 = np.arange(q)[np.asarray([ci[0]>0  for ci in c])] 

t = np.array([4/9, 1/9, 1/9, 1/9, 1/36, 1/36, 1/9, 1/36, 1/36])


def equilibrium(rho,u):             
    cu   = np.dot(c,u.transpose(1,0,2))
    usqr = u[0]**2+u[1]**2
    feq  = np.zeros((q,nx,ny))
    for i in range(q): 
        feq[i,:,:] = rho*t[i] + rho*t[i]*( 3*cu[i] + (9/2)*cu[i]**2 - (3/2)*usqr )
    return feq

feq = equilibrium(1.0,vel)
fin = feq.copy()


timel = np.zeros(maxIter//100)
iterl = np.zeros(maxIter//100)
start = timeit.default_timer()
for Iter in range(maxIter):

    fin[i1,-1,:] = fin[i1,-2,:]      

    rho = sum(fin)  
    u   = np.dot(c.transpose(), fin.transpose((1,0,2)))/rho    

    u[:,0,:] = vel[:,0,:] 

    rho[0,:] = 1/(1-u[0,0,:]) * (sum(fin[i2,0,:])+2.*sum(fin[i1,0,:]))

    feq = equilibrium(rho,u) 

    fin[i3,0,:] = fin[i1,0,:] + feq[i3,0,:] - fin[i1,0,:]

    fout = fin - omega * (fin - feq)

    noslip = [c.tolist().index((-c[i]).tolist()) for i in range(q)]

    for i in range(q): 
        fout[i,obstacle] = fin[noslip[i],obstacle]

    for i in range(q): 
        amat = np.roll(fout[i,:,:],c[i,0],axis=0)
        fin[i,:,:] = np.roll(amat,c[i,1],axis=1)

    if (Iter%100==0): 
#        plt.clf(); 
#        plt.imshow(np.sqrt(u[0]**2+u[1]**2).transpose(),cmap=cm.YlGn)
        print(Iter)
#        #plt.savefig("vel."+str(Iter/100).zfill(4)+".png", bbox_inches='tight')
        stop = timeit.default_timer()
        T = stop-start
        print(T)
        timel[Iter//100]=T
        iterl[Iter//100]=Iter
print(iterl)
print(timel)
# Applying a linear fit with polyfit
b, m = polyfit(iterl, timel, 1)
plt.xlabel('iterations')
plt.ylabel('time (s)')
plt.plot(iterl, timel, '.')
plt.plot(iterl, b + m * iterl, '-')
plt.savefig('timevsiter.png', bbox_inches='tight', dpi=200)



