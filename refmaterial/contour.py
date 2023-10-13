import numpy as np
from math import sqrt
from matplotlib import pyplot as plt, cm
import os

#T = 400
#MC = 200

fspath = input("Enter directory name:\n")

fpath = "tab/" + fspath
all_files = os.scandir(fpath)

contour_matrix = np.zeros((3,5))
for file in all_files:
    with open(file,"r") as f:
        fname = f.name
        quantum = np.genfromtxt(fname,delimiter=',')
        #quantum = quantum[:,25:]
        MC = quantum.shape[0]
        T = quantum.shape[1]
        superposition = np.mean(quantum,axis=0)
        supersd = np.std(quantum,axis=0)
        supererr = supersd*2.326/sqrt(MC)
        fname = fname.replace(fpath,"").replace(".csv","").replace("/","").replace("..","")
        i = 0
        if "UBAR0.7" in fname and "VBAR0.7" in fname:
            i = 0
        elif "UBAR0.7" in fname and "VBAR1.0" in fname:
            i = 1
        elif "UBAR1.0" in fname:
            i = 2

        j = 0
        if "WF[1.0" in fname:
            j = 0
        elif "WF[0.7" in fname:
            j = 1
        elif "WF[0.5" in fname:
            j = 2
        elif "WF[0.2" in fname:
            j = 3
        elif "WF[0.0" in fname:
            j = 4

        contour_matrix[i,j] = T - np.sum(superposition)
        #print(T - np.sum(superposition))


CM = plt.contourf(contour_matrix, vmin=60, vmax=210, cmap=cm.Greys)#, levels=12)
plt.yticks([0,1,2],["High\nsocial\nnorms\n\n\n", "Moderate\nsocial\nnorms", "No\nsocial\nnorms"], fontsize=14)
plt.xticks([0,1,2,3,4],["No\nteam-based\nincentives", "Low\nteam-based\nincentives", "Moderate\nteam-based\nincentives", "High\nteam-based\nincentives", "Full\nteam-based\nincentives"], fontsize=14)
plt.contour(contour_matrix,colors="black",linewidths=0.1)#,levels=12)
#plt.colorbar(CM)
fname = "fig/" + fspath + "_contour.pdf"
plt.savefig(fname,bbox_inches="tight")
print(contour_matrix)
