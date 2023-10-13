import numpy as np
from scipy.optimize import linprog


A = np.array([[3000,800,250],
             [1000,500,200],
             [1   ,0  ,0  ]])

b = np.array([25000,30000,10])

x = [9,1,1]

def valuer(xx,aa,bb):
    ref = np.dot(aa,xx) - bb
    dplus = [np.max((0,z)) for z in ref]
    dminus = [np.min((0,z)) for z in ref]
    out = sum(dplus) + sum(dminus)
    return out

valuer(x,A,b)
