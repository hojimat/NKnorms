import logging
from pathlib import Path
import json
import itertools
import numpy as np
from matplotlib import pyplot as plt

def plot(dir_path):
    """Function that plots all files in a directory as curves in a single graph"""

    for file in sorted(dir_path.iterdir(), reverse=True):
        # parse file and get data
        with open(file, "r") as f:
            fname = f.name
            quantum = np.genfromtxt(fname,delimiter=',')
            MC, T = quantum.shape

        # analyze data
        superposition = np.mean(quantum, axis=0)
        supersd = np.std(quantum,axis=0)
        supererr = supersd*2.5758/np.sqrt(MC)

        # generate labels
        fname = fname.replace(str(dir_path),"").replace(".csv","").replace("/","").replace("..","")

        # plot the curve
        plt.plot(list(range(T)),superposition, label=fname, linewidth=1.0)

        # plot confidence intervals
        plt.fill_between(list(range(T)),superposition-supererr,superposition+supererr,
                        alpha=0.1)

        #plt.ylim(0.1, 1.0)
        plt.grid(True)
   
    # close the plot
    plt.legend(loc="lower right",prop={'family':'serif','size':5})
    fname = "fig/" + str(dir_path).replace("/","") + ".pdf"
    plt.savefig(fname, bbox_inches="tight")
    plt.clf()
    logging.info(f"done: {fname}")

def main():
    """Main function"""

    logging.basicConfig(level=logging.INFO)
    with open('structure.json', 'r', encoding='utf-8') as file:
        data = json.load(file)  
        structure = data['structure']
        patterns = data['patterns']
    

    for path in itertools.product(*structure):
        p = Path(*path)
        plot(p)


if __name__=='__main__':
    main()
