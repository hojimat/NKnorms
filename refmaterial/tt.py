import numpy as np
from time import time
start = time()
def hel(z):
    tmp = np.random.choice(z,2)
    return tmp
a = np.arange(16)
[hel(a[z*4:(z+1)*4]) for z in range(4)]
end = time()
print(end-start)

start = time()
def hel(z):
    tmp = np.random.choice(z,2)
    return tmp
a = np.arange(16).reshape(4,4)
np.apply_along_axis(hel,0,a)
end = time()
print(end-start)

