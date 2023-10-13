import numpy as np
from math import sqrt
from matplotlib import pyplot as plt
import os

#T = 400
#MC = 200

fspath = input("Enter directory name:\n")

fpath = "tab/" + fspath
all_files = os.scandir(fpath)

for file in all_files:
    with open(file,"r") as f:
        fname = f.name
        quantum = np.genfromtxt(fname,delimiter=',')
        MC = quantum.shape[0]
        T = quantum.shape[1]
        superposition = np.mean(quantum,axis=0)
        supersd = np.std(quantum,axis=0)
        supererr = supersd*2.326/sqrt(MC)
        fname = fname.replace(fpath,"").replace(".csv","").replace("/","").replace("..","")
        plt.plot(list(range(T)),superposition,label=fname)
        plt.fill_between(list(range(T)),superposition-supererr,superposition+supererr,alpha=0.5)
        plt.legend(prop={'size': 5})
fname = "fig/" + fspath + ".pdf"
plt.savefig(fname)
        #plt.close()


