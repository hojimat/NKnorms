import numpy as np
from matplotlib import pyplot as plt
from scipy.stats import norm

N = 15
P = 4
rho = 0.3
corrmat = np.repeat(rho,P*P).reshape(P,P) + (1-rho) * np.eye(P)
cholesky = np.linalg.cholesky(corrmat)
cholesky = (cholesky.T / np.sum(cholesky,axis=1)).T

eps = np.random.normal(0,1,(2**N)*P).reshape(2**N,P)
X = np.dot(cholesky,eps.T).T


print(np.corrcoef(X[:,0],X[:,1]))
print(np.corrcoef(X[:,0],X[:,2]))
print(np.corrcoef(X[:,0],X[:,3]))
print(np.corrcoef(X[:,1],X[:,3]))
print(np.corrcoef(X[:,2],X[:,3]))

X = norm.cdf(X)
print(np.corrcoef(X[:,0],X[:,1]))
print(np.corrcoef(X[:,0],X[:,2]))
print(np.corrcoef(X[:,0],X[:,3]))
print(np.corrcoef(X[:,1],X[:,3]))
print(np.corrcoef(X[:,2],X[:,3]))

for i in range(P):
    plt.hist(X[:,i])
    plt.show()
