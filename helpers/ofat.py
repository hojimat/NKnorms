"""
Performs OFAT sensitivity

The process flow is as follows:
   main() -> 
   get_all_outcomes() ->
   get_simrun_outcome() ->
   Nature() ->
   summarize() ->

"""

import logging
from multiprocessing import Pool
import numpy as np
import progressbar
from models import Nature
import nkpack as nk


MC = 500
PARAMS_SET = (
# defaults
{"p":5,"n":4,"kcs":(3,0,0),"t":500,"rho":0.9,"nsoc":4,"deg":2,"net":3,"xi":1.0,"tm":50,"coord":2,"apc":(2,2,4),"wf":1.0,"goals":(1.,1.),"w":0.5,"normalize":True,"precompute":True}, #hierar224
{"p":5,"n":4,"kcs":(3,0,0),"t":500,"rho":0.9,"nsoc":4,"deg":2,"net":3,"xi":1.0,"tm":50,"coord":0,"apc":(2,1,1),"wf":1.0,"goals":(1.,1.),"w":0.5,"normalize":True,"precompute":True}, #decent211



# variate P
{"p":10,"n":4,"kcs":(3,0,0),"t":500,"rho":0.9,"nsoc":4,"deg":2,"net":3,"xi":1.0,"tm":50,"coord":2,"apc":(2,2,4),"wf":1.0,"goals":(1.,1.),"w":0.5,"normalize":False,"precompute":False}, #hierar224 p10
{"p":20,"n":4,"kcs":(3,0,0),"t":500,"rho":0.9,"nsoc":4,"deg":2,"net":3,"xi":1.0,"tm":50,"coord":2,"apc":(2,2,4),"wf":1.0,"goals":(1.,1.),"w":0.5,"normalize":False,"precompute":False}, #hierar224 p20
{"p":50,"n":4,"kcs":(3,0,0),"t":500,"rho":0.9,"nsoc":4,"deg":2,"net":3,"xi":1.0,"tm":50,"coord":2,"apc":(2,2,4),"wf":1.0,"goals":(1.,1.),"w":0.5,"normalize":False,"precompute":False}, #hierar224 p50
{"p":100,"n":4,"kcs":(3,0,0),"t":500,"rho":0.9,"nsoc":4,"deg":2,"net":3,"xi":1.0,"tm":50,"coord":2,"apc":(2,2,4),"wf":1.0,"goals":(1.,1.),"w":0.5,"normalize":False,"precompute":False}, #hierar224 p100
{"p":10,"n":4,"kcs":(3,0,0),"t":500,"rho":0.9,"nsoc":4,"deg":2,"net":3,"xi":1.0,"tm":50,"coord":2,"apc":(2,2,4),"wf":1.0,"goals":(1.,1.),"w":0.5,"normalize":False,"precompute":False}, #decent211 p10
{"p":20,"n":4,"kcs":(3,0,0),"t":500,"rho":0.9,"nsoc":4,"deg":2,"net":3,"xi":1.0,"tm":50,"coord":2,"apc":(2,2,4),"wf":1.0,"goals":(1.,1.),"w":0.5,"normalize":False,"precompute":False}, #decent211 p20
{"p":50,"n":4,"kcs":(3,0,0),"t":500,"rho":0.9,"nsoc":4,"deg":2,"net":3,"xi":1.0,"tm":50,"coord":2,"apc":(2,2,4),"wf":1.0,"goals":(1.,1.),"w":0.5,"normalize":False,"precompute":False}, #decent211 p50
{"p":100,"n":4,"kcs":(3,0,0),"t":500,"rho":0.9,"nsoc":4,"deg":2,"net":3,"xi":1.0,"tm":50,"coord":2,"apc":(2,2,4),"wf":1.0,"goals":(1.,1.),"w":0.5,"normalize":False,"precompute":False}, #decent211 p100



# variate RHO
{"p":5,"n":4,"kcs":(3,0,0),"t":500,"rho":1.0,"nsoc":4,"deg":2,"net":3,"xi":1.0,"tm":50,"coord":2,"apc":(2,2,4),"wf":1.0,"goals":(1.,1.),"w":0.5,"normalize":True,"precompute":True}, #hierar224
{"p":5,"n":4,"kcs":(3,0,0),"t":500,"rho":0.6,"nsoc":4,"deg":2,"net":3,"xi":1.0,"tm":50,"coord":2,"apc":(2,2,4),"wf":1.0,"goals":(1.,1.),"w":0.5,"normalize":True,"precompute":True}, #hierar224
{"p":5,"n":4,"kcs":(3,0,0),"t":500,"rho":0.3,"nsoc":4,"deg":2,"net":3,"xi":1.0,"tm":50,"coord":2,"apc":(2,2,4),"wf":1.0,"goals":(1.,1.),"w":0.5,"normalize":True,"precompute":True}, #hierar224
{"p":5,"n":4,"kcs":(3,0,0),"t":500,"rho":1.0,"nsoc":4,"deg":2,"net":3,"xi":1.0,"tm":50,"coord":0,"apc":(2,1,1),"wf":1.0,"goals":(1.,1.),"w":0.5,"normalize":True,"precompute":True}, #decent211
{"p":5,"n":4,"kcs":(3,0,0),"t":500,"rho":0.6,"nsoc":4,"deg":2,"net":3,"xi":1.0,"tm":50,"coord":0,"apc":(2,1,1),"wf":1.0,"goals":(1.,1.),"w":0.5,"normalize":True,"precompute":True}, #decent211
{"p":5,"n":4,"kcs":(3,0,0),"t":500,"rho":0.3,"nsoc":4,"deg":2,"net":3,"xi":1.0,"tm":50,"coord":0,"apc":(2,1,1),"wf":1.0,"goals":(1.,1.),"w":0.5,"normalize":True,"precompute":True}, #decent211



# variate TM
{"p":5,"n":4,"kcs":(3,0,0),"t":500,"rho":0.9,"nsoc":4,"deg":2,"net":3,"xi":1.0,"tm":25,"coord":2,"apc":(2,2,4),"wf":1.0,"goals":(1.,1.),"w":0.5,"normalize":True,"precompute":True}, #hierar224
{"p":5,"n":4,"kcs":(3,0,0),"t":500,"rho":0.9,"nsoc":4,"deg":2,"net":3,"xi":1.0,"tm":75,"coord":2,"apc":(2,2,4),"wf":1.0,"goals":(1.,1.),"w":0.5,"normalize":True,"precompute":True}, #hierar224
{"p":5,"n":4,"kcs":(3,0,0),"t":500,"rho":0.9,"nsoc":4,"deg":2,"net":3,"xi":1.0,"tm":100,"coord":2,"apc":(2,2,4),"wf":1.0,"goals":(1.,1.),"w":0.5,"normalize":True,"precompute":True}, #hierar224
{"p":5,"n":4,"kcs":(3,0,0),"t":500,"rho":0.9,"nsoc":4,"deg":2,"net":3,"xi":1.0,"tm":25,"coord":0,"apc":(2,1,1),"wf":1.0,"goals":(1.,1.),"w":0.5,"normalize":True,"precompute":True}, #decent211
{"p":5,"n":4,"kcs":(3,0,0),"t":500,"rho":0.9,"nsoc":4,"deg":2,"net":3,"xi":1.0,"tm":75,"coord":0,"apc":(2,1,1),"wf":1.0,"goals":(1.,1.),"w":0.5,"normalize":True,"precompute":True}, #decent211
{"p":5,"n":4,"kcs":(3,0,0),"t":500,"rho":0.9,"nsoc":4,"deg":2,"net":3,"xi":1.0,"tm":100,"coord":0,"apc":(2,1,1),"wf":1.0,"goals":(1.,1.),"w":0.5,"normalize":True,"precompute":True}, #decent211


# variate WF
{"p":5,"n":4,"kcs":(3,0,0),"t":500,"rho":0.9,"nsoc":4,"deg":2,"net":3,"xi":1.0,"tm":50,"coord":2,"apc":(2,2,4),"wf":0.75,"goals":(1.,1.),"w":0.5,"normalize":True,"precompute":True}, #hierar224
{"p":5,"n":4,"kcs":(3,0,0),"t":500,"rho":0.9,"nsoc":4,"deg":2,"net":3,"xi":1.0,"tm":50,"coord":2,"apc":(2,2,4),"wf":0.5,"goals":(1.,1.),"w":0.5,"normalize":True,"precompute":True}, #hierar224
{"p":5,"n":4,"kcs":(3,0,0),"t":500,"rho":0.9,"nsoc":4,"deg":2,"net":3,"xi":1.0,"tm":50,"coord":2,"apc":(2,2,4),"wf":0.25,"goals":(1.,1.),"w":0.5,"normalize":True,"precompute":True}, #hierar224
{"p":5,"n":4,"kcs":(3,0,0),"t":500,"rho":0.9,"nsoc":4,"deg":2,"net":3,"xi":1.0,"tm":50,"coord":0,"apc":(2,1,1),"wf":0.75,"goals":(1.,1.),"w":0.5,"normalize":True,"precompute":True}, #decent211
{"p":5,"n":4,"kcs":(3,0,0),"t":500,"rho":0.9,"nsoc":4,"deg":2,"net":3,"xi":1.0,"tm":50,"coord":0,"apc":(2,1,1),"wf":0.5,"goals":(1.,1.),"w":0.5,"normalize":True,"precompute":True}, #decent211
{"p":5,"n":4,"kcs":(3,0,0),"t":500,"rho":0.9,"nsoc":4,"deg":2,"net":3,"xi":1.0,"tm":50,"coord":0,"apc":(2,1,1),"wf":0.25,"goals":(1.,1.),"w":0.5,"normalize":True,"precompute":True}, #decent211



# variate W
{"p":5,"n":4,"kcs":(3,0,0),"t":500,"rho":0.9,"nsoc":4,"deg":2,"net":3,"xi":1.0,"tm":50,"coord":2,"apc":(2,2,4),"wf":1.0,"goals":(1.,1.),"w":1.0,"normalize":True,"precompute":True}, #hierar224
{"p":5,"n":4,"kcs":(3,0,0),"t":500,"rho":0.9,"nsoc":4,"deg":2,"net":3,"xi":1.0,"tm":50,"coord":2,"apc":(2,2,4),"wf":1.0,"goals":(1.,1.),"w":0.75,"normalize":True,"precompute":True}, #hierar224
{"p":5,"n":4,"kcs":(3,0,0),"t":500,"rho":0.9,"nsoc":4,"deg":2,"net":3,"xi":1.0,"tm":50,"coord":2,"apc":(2,2,4),"wf":1.0,"goals":(1.,1.),"w":0.25,"normalize":True,"precompute":True}, #hierar224
{"p":5,"n":4,"kcs":(3,0,0),"t":500,"rho":0.9,"nsoc":4,"deg":2,"net":3,"xi":1.0,"tm":50,"coord":2,"apc":(2,2,4),"wf":1.0,"goals":(1.,1.),"w":0.0,"normalize":True,"precompute":True}, #hierar224
{"p":5,"n":4,"kcs":(3,0,0),"t":500,"rho":0.9,"nsoc":4,"deg":2,"net":3,"xi":1.0,"tm":50,"coord":0,"apc":(2,1,1),"wf":1.0,"goals":(1.,1.),"w":1.0,"normalize":True,"precompute":True}, #decent211
{"p":5,"n":4,"kcs":(3,0,0),"t":500,"rho":0.9,"nsoc":4,"deg":2,"net":3,"xi":1.0,"tm":50,"coord":0,"apc":(2,1,1),"wf":1.0,"goals":(1.,1.),"w":0.75,"normalize":True,"precompute":True}, #decent211
{"p":5,"n":4,"kcs":(3,0,0),"t":500,"rho":0.9,"nsoc":4,"deg":2,"net":3,"xi":1.0,"tm":50,"coord":0,"apc":(2,1,1),"wf":1.0,"goals":(1.,1.),"w":0.25,"normalize":True,"precompute":True}, #decent211
{"p":5,"n":4,"kcs":(3,0,0),"t":500,"rho":0.9,"nsoc":4,"deg":2,"net":3,"xi":1.0,"tm":50,"coord":0,"apc":(2,1,1),"wf":1.0,"goals":(1.,1.),"w":0.0,"normalize":True,"precompute":True}, #decent211



# variate GOAL_PERF
{"p":5,"n":4,"kcs":(3,0,0),"t":500,"rho":0.9,"nsoc":4,"deg":2,"net":3,"xi":1.0,"tm":50,"coord":2,"apc":(2,2,4),"wf":1.0,"goals":(0.8,1.),"w":0.5,"normalize":True,"precompute":True}, #hierar224
{"p":5,"n":4,"kcs":(3,0,0),"t":500,"rho":0.9,"nsoc":4,"deg":2,"net":3,"xi":1.0,"tm":50,"coord":2,"apc":(2,2,4),"wf":1.0,"goals":(0.6,1.),"w":0.5,"normalize":True,"precompute":True}, #hierar224
{"p":5,"n":4,"kcs":(3,0,0),"t":500,"rho":0.9,"nsoc":4,"deg":2,"net":3,"xi":1.0,"tm":50,"coord":0,"apc":(2,1,1),"wf":1.0,"goals":(0.8,1.),"w":0.5,"normalize":True,"precompute":True}, #decent211
{"p":5,"n":4,"kcs":(3,0,0),"t":500,"rho":0.9,"nsoc":4,"deg":2,"net":3,"xi":1.0,"tm":50,"coord":0,"apc":(2,1,1),"wf":1.0,"goals":(0.6,1.),"w":0.5,"normalize":True,"precompute":True}, #decent211


# variate GOAL_SYNC
{"p":5,"n":4,"kcs":(3,0,0),"t":500,"rho":0.9,"nsoc":4,"deg":2,"net":3,"xi":1.0,"tm":50,"coord":2,"apc":(2,2,4),"wf":1.0,"goals":(1.,0.8),"w":0.5,"normalize":True,"precompute":True}, #hierar224
{"p":5,"n":4,"kcs":(3,0,0),"t":500,"rho":0.9,"nsoc":4,"deg":2,"net":3,"xi":1.0,"tm":50,"coord":2,"apc":(2,2,4),"wf":1.0,"goals":(1.,0.6),"w":0.5,"normalize":True,"precompute":True}, #hierar224
{"p":5,"n":4,"kcs":(3,0,0),"t":500,"rho":0.9,"nsoc":4,"deg":2,"net":3,"xi":1.0,"tm":50,"coord":0,"apc":(2,1,1),"wf":1.0,"goals":(1.,0.8),"w":0.5,"normalize":True,"precompute":True}, #decent211
{"p":5,"n":4,"kcs":(3,0,0),"t":500,"rho":0.9,"nsoc":4,"deg":2,"net":3,"xi":1.0,"tm":50,"coord":0,"apc":(2,1,1),"wf":1.0,"goals":(1.,0.6),"w":0.5,"normalize":True,"precompute":True}, #decent211


# variate P
{"p":10,"n":4,"kcs":(3,0,0),"t":500,"rho":0.9,"nsoc":4,"deg":2,"net":3,"xi":1.0,"tm":50,"coord":2,"apc":(2,2,4),"wf":1.0,"goals":(1.,1.),"w":0.5,"normalize":False,"precompute":False}, #hierar224 p10
{"p":20,"n":4,"kcs":(3,0,0),"t":500,"rho":0.9,"nsoc":4,"deg":2,"net":3,"xi":1.0,"tm":50,"coord":2,"apc":(2,2,4),"wf":1.0,"goals":(1.,1.),"w":0.5,"normalize":False,"precompute":False}, #hierar224 p20
{"p":50,"n":4,"kcs":(3,0,0),"t":500,"rho":0.9,"nsoc":4,"deg":2,"net":3,"xi":1.0,"tm":50,"coord":2,"apc":(2,2,4),"wf":1.0,"goals":(1.,1.),"w":0.5,"normalize":False,"precompute":False}, #hierar224 p50
{"p":100,"n":4,"kcs":(3,0,0),"t":500,"rho":0.9,"nsoc":4,"deg":2,"net":3,"xi":1.0,"tm":50,"coord":2,"apc":(2,2,4),"wf":1.0,"goals":(1.,1.),"w":0.5,"normalize":False,"precompute":False}, #hierar224 p100
{"p":10,"n":4,"kcs":(3,0,0),"t":500,"rho":0.9,"nsoc":4,"deg":2,"net":3,"xi":1.0,"tm":50,"coord":2,"apc":(2,2,4),"wf":1.0,"goals":(1.,1.),"w":0.5,"normalize":False,"precompute":False}, #decent211 p10
{"p":20,"n":4,"kcs":(3,0,0),"t":500,"rho":0.9,"nsoc":4,"deg":2,"net":3,"xi":1.0,"tm":50,"coord":2,"apc":(2,2,4),"wf":1.0,"goals":(1.,1.),"w":0.5,"normalize":False,"precompute":False}, #decent211 p20
{"p":50,"n":4,"kcs":(3,0,0),"t":500,"rho":0.9,"nsoc":4,"deg":2,"net":3,"xi":1.0,"tm":50,"coord":2,"apc":(2,2,4),"wf":1.0,"goals":(1.,1.),"w":0.5,"normalize":False,"precompute":False}, #decent211 p50
{"p":100,"n":4,"kcs":(3,0,0),"t":500,"rho":0.9,"nsoc":4,"deg":2,"net":3,"xi":1.0,"tm":50,"coord":2,"apc":(2,2,4),"wf":1.0,"goals":(1.,1.),"w":0.5,"normalize":False,"precompute":False}, #decent211 p100

)


def run_simulation(prms, bar_, mc_):
    """Run a single simulation give params"""

    nature = Nature(**prms)
    np.random.seed()
    nature.initialize()
    nature.play()
    perfs = nature.organization.performances.mean(axis=1)
    syncs = nature.organization.synchronies
    return perfs, syncs


if __name__=="__main__":
    logging.basicConfig(level=logging.WARNING)
    for params in PARAMS_SET:
        pbar = progressbar.ProgressBar(max_value=MC)

        def worker(i):
            """A worker function that calls iterations in a Pool"""
            return run_simulation(params, pbar, i)

        pbar.start()
        with Pool(8) as pool:
            quantum = pool.map(worker, range(MC))
        pbar.finish()

        # T x MC array of mean performance and synchrony of an
        # organization at every period for MC repetitions
        performances = [z[0] for z in quantum]
        synchronies = [z[1] for z in quantum]

        # save to files
        params_filename = "".join(f"{k}{v}" for k,v in params.items())
        params_filename = params_filename.replace(" ", "") + ".csv"

        np.savetxt("sens/perf/" + params_filename, performances, delimiter=',', fmt='%10.5f')
        np.savetxt("sens/sync/" + params_filename, synchronies, delimiter=',', fmt='%10.5f')

    
    print("Done")
