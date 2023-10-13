import numpy as np
from scipy.stats import beta
from math import sqrt
import matplotlib.pyplot as plt
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
    plt.savefig(f"fig/spenser_{var}.pdf")
    if fig==True:
        plt.show()



xMC = 500
xN = 4
xpub = 2
xK = 1
xT = 300
xTh = 2
xnay = 1

bar = progressbar.ProgressBar(max_value=xMC)

# def single_iteration(y):
# 	a = Landscape(xN,xK,xT,xnay)
# 	a.initialize()
# 	a.simulation()
# 	poker = a.contrib_space/a.perfmax
# 	bar.update(y)
# 	return poker

def single_iteration(y):
    a = Landscape(xN+xpub,xK,xT,xnay)
    b = Landscape(xN+xpub,xK,xT,xnay)
    c = Landscape(xN+xpub,xK,xT,xnay)
    d = Landscape(xN+xpub,xK,xT,xnay)

    a.initialize()
    b.initialize()
    c.initialize()
    d.initialize()


    for t in range(1,xT):
        tmp_a = a.decision_space[t-1,:].copy()
        bmp_a = a.betas[t-1].copy()
        tmp_b = b.decision_space[t-1,:].copy()
        bmp_b = b.betas[t-1].copy()
        tmp_c = c.decision_space[t-1,:].copy()
        bmp_c = c.betas[t-1].copy()
        tmp_d = d.decision_space[t-1,:].copy()
        bmp_d = d.betas[t-1].copy()

        force_a = None
        force_b = None
        force_c = None
        force_d = None
        if t < xTh:
            force_a = nk.reinforcement_climb(tmp_a,bmp_a,a.nay,a.perfmat)
            force_b = nk.reinforcement_climb(tmp_b,bmp_b,b.nay,b.perfmat)
            force_c = nk.reinforcement_climb(tmp_c,bmp_c,c.nay,c.perfmat)
            force_d = nk.reinforcement_climb(tmp_d,bmp_d,d.nay,d.perfmat)
        else:
            force_a = nk.descriptive_climb(tmp_a,bmp_a,a.nay,a.perfmat,normlib)
            force_b = nk.descriptive_climb(tmp_b,bmp_b,b.nay,b.perfmat,normlib)
            force_c = nk.descriptive_climb(tmp_c,bmp_c,c.nay,c.perfmat,normlib)
            force_d = nk.descriptive_climb(tmp_d,bmp_d,d.nay,d.perfmat,normlib)
        #end
        a.decision_space[t,:] = force_a[0]
        a.contrib_space[t] = force_a[1]
        a.betas[t] = force_a[2]
        b.decision_space[t,:] = force_b[0]
        b.contrib_space[t] = force_b[1]
        b.betas[t] = force_b[2]
        c.decision_space[t,:] = force_c[0]
        c.contrib_space[t] = force_c[1]
        c.betas[t] = force_c[2]
        d.decision_space[t,:] = force_d[0]
        d.contrib_space[t] = force_d[1]
        d.betas[t] = force_d[2]
        a.share_public_bits()
        b.share_public_bits()
        c.share_public_bits()
        d.share_public_bits()
    #end



if __name__ == '__main__':
 pool = Pool(3)
 monaco = []
 monaco.append(pool.map(single_iteration,range(xMC)))
 pool.close()
 bar.finish()
 monaco = np.array(monaco)[0]
 meanaco = np.mean(monaco,axis=0)
 sdaco = np.std(monaco,axis=0)
 erraco = sdaco*1.96/sqrt(xMC)
 plt.plot(list(range(xT)),meanaco, label=f"phi")
 plt.fill_between(list(range(xT)),meanaco-erraco,meanaco+erraco, alpha=0.5)
 plt.legend()
 #plt.savefig(f"K={xK}.pdf")
 plt.show()
