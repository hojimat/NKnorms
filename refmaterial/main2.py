import numpy as np
from scipy.stats import beta
import matplotlib.pyplot as plt
from math import sqrt
import NKPackage as nk
import progressbar
from thesettings import Landscape
from multiprocessing import Pool, Process

def sensitivity_analysis(N,K,T,nay,MC,var,obj,fig=True):
    senser = []
    if var == "N":
        senser = [[z,K,T,nay] for z in [4,7,10,13,15,20]]
    elif var == "K":
        senser = [[N,z,T,nay] for z in [0,1,3,5,8]]
    elif var == "nay":
        senser = [[N,K,T,z] for z in [1,2,3,4,5,6,7,8,9]]
    elif var is None:
        senser = [[N,K,T,nay]]

    for sense in senser:
        print(sense)
        monaco = np.zeros([MC,T])
        bar = progressbar.ProgressBar(max_value=MC)
        for mc in range(MC):
            a = obj(*sense)
            a.initialize()
            a.simulation()
            monaco[mc,:] = a.contrib_space / a.perfmax
            bar.update(mc)
        meanaco = np.mean(monaco,axis=0)
        sdaco = np.std(monaco,axis=0)
        uco = meanaco + sdaco
        dco = meanaco - sdaco
        plt.plot(list(range(T)),meanaco, label=f"{var}={sense[1]}")
        #plt.plot(list(range(T)),uco,linestyle="--",color="gray")
        #plt.plot(list(range(T)),dco,linestyle="--",color="gray")

    plt.legend()
    plt.savefig(f"fig/senser_{var}.pdf")
    if fig==True:
        plt.show()



xMC = 5000
xN = 10
xK = 3
xT = 100
xnay = 2
bar = progressbar.ProgressBar(max_value=xMC)

def single_iteration(i):
	a = Landscape(xN,xK,xT,xnay)
	a.initialize()
	a.simulation()
	poker = a.contrib_space/a.perfmax
	bar.update(i)
	return poker 

if __name__ == '__main__':
    for i in [0,1,5,9]:
        monaco = []
        xK = i
        print(xK)
        pool = Pool(4)
        monaco.append(pool.map(single_iteration,range(xMC)))
        pool.close()
        bar.finish()
        monaco = np.array(monaco)[0]
        meanaco = np.mean(monaco,axis=0)
        sdaco = np.std(monaco,axis=0)
        erraco = sdaco*1.96/sqrt(xMC)
        plt.plot(list(range(xT)),meanaco, label=f"K={i}")
        plt.fill_between(list(range(xT)),meanaco-erraco,meanaco+erraco)

    plt.legend()
    plt.savefig(f"recent.pdf")
    plt.show()


