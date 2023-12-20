import numpy as np
from math import sqrt
from matplotlib import pyplot as plt
import os

#T = 400
#MC = 200

#fspath = input("Enter directory name:\n")

def mainplot(fspath):
    fpath = fspath
    all_files = sorted(os.scandir(fpath),key=lambda e: e.name,reverse=True)
    for file in all_files:
        with open(file,"r") as f:
            fname = f.name
            quantum = np.genfromtxt(fname,delimiter=',')[:,50:]
            MC = quantum.shape[0]
            T = quantum.shape[1]
            superposition = np.mean(quantum,axis=0)
            supersd = np.std(quantum,axis=0)
            supererr = supersd*2.5758/sqrt(MC)
            fname = fname.replace(fpath,"").replace(".csv","").replace("/","").replace("..","")
            plt.plot(list(range(T)),superposition,
                    label=fname,
                    linewidth=1.0,
                    )

            plt.fill_between(list(range(T)),superposition-supererr,superposition+supererr,
                            alpha=0.1)
    fname = "fig/" + fspath.replace("/","") + ".pdf"
    #plt.legend(loc="lower right",prop={'family':'serif','size':10},bbox_to_anchor=(0,1))
    plt.legend(loc="lower right",prop={'family':'serif','size':5})
    #plt.ylim(0.7,0.95)
    plt.savefig(fname,bbox_inches="tight")
    plt.clf()

dir0 = "aperf"
dirnames = []
for w in ["w1","w7","w5","w0"]:
    for rho in ["rho3","rho9"]:
        for kcs in ["kcs300", "kcs121", "kcs331", "kcs222"]:
            dirnames.append(f"{dir0}/{w}/{rho}/{kcs}/")

for fff in dirnames:
    mainplot(fff)
    print(fff)
