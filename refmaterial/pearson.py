import numpy as np
from scipy.stats import norm
from matplotlib import pyplot as plt

N = 4
P = 4
rho = 0.1

corrmat = np.repeat(rho,P*P).reshape(P,P) + (1-rho) * np.eye(P)
corrmat = 2*np.sin((np.pi / 6 ) * corrmat) #* 144
X = np.random.multivariate_normal(mean=[0]*P,cov=corrmat,size=4)
X = norm.cdf(X)#.reshape(5,2)

#print(X[1:5,:])
#print(np.corrcoef(X[:,1],X[:,3]))
print(X)
print("-----------------")

#fnc = lambda z: z.reshape(2,2)
#X = np.apply_along_axis(fnc,0,X)
#print(X)

X = np.reshape(X.T,(8,2)).T
#X = np.reshape(X,(2,8))
print(X)
print("-----------------")
