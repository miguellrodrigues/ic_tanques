#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Oct 17 10:56:46 2022

@author: lqsoliveira
"""

from asyncore import read
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
plt.close('all')

#Read files CSV
read_u = pd.read_csv('U.csv', sep = ',', header=None)
read_H3 = pd.read_csv('H3.csv', sep = ',', header=None)
read_H4 = pd.read_csv('H4.csv', sep = ',', header=None)

#Variáveis Reais
H3_r = read_H3[:][0]
H4_r = read_H4[:][0]
u_r = read_u[:][0]

t = np.arange(0,len(H3_r),1)

h3 = np.empty(len(t))
h3.fill(np.nan)
h4 = np.empty(len(t))
h4.fill(np.nan)

r = 31
mu = 40
sigma = 55
a4 = 3019

h3[0]=0
h4[0]=0

for i in range(0,len(t)-1,1):
    if h3[i] < 0:
        h3[i] = 0

    q34 = (29.4*(h4[i]-h3[i]) - 83.93)
    qout = (165.48*np.sqrt(h3[i]) - 147.01)
    qin = (16.46*u_r[i] - 156.93)
    #inicio calculo area t3

    z1 = np.sqrt(h3[i])
    z2 = np.cos(2.5*np.pi * (h4[i] - mu)) / (sigma * np.sqrt(2 * np.pi))
    z3 = np.exp(-((h4[i] - mu)**2) / (2 * sigma**2))

    a3 = ((3*r)/5) * (2.7*r - (z2 * z3))

    #termino calculo area T3
    h3[i+1] = h3[i] + 1*((q34-.965*qout)/a3)
    h4[i+1] = h4[i] + 1*((qin-q34)/a4)


plt.figure()
plt.subplot(3,1,1)
plt.plot(t,H3_r,'b')
plt.plot(t,h3,'r')
plt.grid()
plt.xlim((0,12500))
plt.subplot(3,1,2)
plt.plot(t,H4_r,'b')
plt.plot(t,h4,'r')
plt.xlim((0,12500))
plt.grid()
plt.subplot(3,1,3)
plt.plot(t,u_r,'k')
plt.xlim((0,12500))
plt.grid()
plt.show()
