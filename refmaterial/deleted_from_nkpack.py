# PURGATORY


def contrib_full(imat,cmat,n,p):
    """depreciated
    Computes performances for all binary vectors of size N*P

    Notes:
        The most processing-heavy part of any simulation. get_global_max() is a 'lazy' alternative

    Args:
        imat (numpy.ndarray): Interaction matrix
        cmat (numpy.ndarray): Contribution matrix
        n (int): Number of tasks per landscape
        p (int): Number of landscapes (population size)

    Returns:
        numpy.ndarray: Array of performances for all vectors of size N*P, normalized by the global maximum
        float: the global maximum value
    """

    n_p = n*p
    perfmat = np.empty((2**n_p,p),dtype=float)
    for i in range(2**n_p):
        bstring = np.array(binx(i,n_p))
        bval = contrib_solve(bstring,imat,cmat,n,p) # !!! processing heavy !!!
        perfmat[i,] = bval
    idxmax = np.argmax(np.mean(perfmat,1))
    perfmax = perfmat[idxmax] 
    
    perfmean = np.mean(perfmat,axis=1)
    perfglobalmax = perfmean.max()
    #perfargmax = perfmean.argmax() 

    output1 = perfmat / perfmax
    #output2 = perfargmax
    #output3 = perfglobalmax
    output3 = perfmax
    return output1, output3#, output2, output3


def artify(n,p,r):
    """depreciated"""

    tmp = np.arange(n*p)
    tmp = tmp.reshape(p,n)
    fnc = lambda z: np.random.choice(z,r,replace=False)
    tmp = np.apply_along_axis(fnc,1,tmp)
    output = tmp
    return output

def calculate_match(x,art):
    """depreciated"""

    if art==[]:
        return 0.0
    tmp = [x[z[0]]==z[1] for z in art]
    output = sum(tmp) / len(tmp) 
    return output


def get_globalmax2(imat,cmat,n,p,brute=True,t0=20,t1=0,alpha=0.1,kk=1):
    """Estimates a global maximum for a landscape using Simulated Annealing algorithm

    Args:
        imat (numpy.ndarray): Interaction matrix
        cmat (numpy.ndarray): Contribution matrix
        n (int): Number of tasks per landscape
        p (int): Number of landscapes (population size)
        t0 (float): Initial temperature. Suggested values are 20 (default) or 1.
        t1 (float): Final temperature. Suggested value is 0 (default)
        alpha (float): Step for temperature. Suggested value are 0.1 (default) or 0.001
        kk (float): Adjustment parameter. Not used at the moment.

    Returns:
        numpy.ndarray: The estimates of the global maximum for each of the P landscapes
    """

    n_p = n*p
    output = None
   
    if brute:
        bstrings = map(lambda x: np.array(binx(x,n_p)), range(2**n_p))
 
        perfmax = [0.0]*p#np.zeros(p,dtype=float)
        for bitstring in bstrings:#range(2**n_p):
            bval = contrib_solve(bitstring,imat,cmat,n,p) # !!! processing heavy !!!
            if np.mean(bval)>np.mean(perfmax):
                perfmax = bval
 
        output = np.array(perfmax)
    else:
        state = np.array(binx(0,n_p))
        value = contrib_solve(state,imat,cmat,n,p)
        
        t = t0
        while t>t1:
            state_ = random_neighbour(state,0,n_p)
            value_ = contrib_solve(state_,imat,cmat,n,p)
            value_ = np.array(value_)
            
            d_mean = np.mean(value_) - np.mean(value)
            #d_sep = value_ - value

            if d_mean > 0 or np.exp(d_mean/t) > np.random.uniform():
                state = state_
                value = value_
            t -= alpha

        output = value
    return output

#def calculate_freq(x,vec):
#    tmp = flatten(vec)
#    if len(tmp)==0:
#        tmp = 0
#    else:
#        tmp = tmp.count(x)/len(tmp)
#    output = tmp
#    return output

def schism(perf1,perf2,social):
    """depreciated"""

    tmp = None
    if social is True:
        tmp = perf2
    else:
        tmp = perf1
    output = tmp
    return output

#def extract_pub(x,myid,n,p,npub):
#    tmp = np.reshape(x,(p,n))
#    tmp = tmp[:,-npub:n]
#    output = tmp
#    return output


def get_neighbours(vec,count):
    """Generates binary vectors that are 1-bit away (a unit Hamming distance)

    Args:
        vec (list or numpy.ndarray): An input vector
        count (int): Number of neighbours to generate

    Returns:
        list: A list of 1-bit neighbours of vec
        list: A list of decimal equivalents of the above
    """

    tmpv = []
    tmpi = []
    subbset = np.random.choice(np.arange(len(vec)),count,replace=False)
    for i in subbset:
        y = vec.copy()
        y[i] = 1 - y[i]
        tmpv.append(y)
        tmpi.append(binx(y))
    return(tmpv, tmpi)

def binx(x,size=4,out=None):
    """Converts values to binary and back

    Args:
        x: Input value (can be any type)
        size (int): Desired output vector size (adds leading zeros if the output size is less than desired, ignores otherwise)
        out (str or None): Specifies output type. Is ignored at the moment.

    Returns:
        list: A list of 0s and 1s if x is int.
        int: A decimal equivalent if x is str, numpy.ndarray, list.
    """

    tmp = x
    output = None
    if type(tmp) is int:
        #tmp = np.binary_repr(tmp,size)
        #tmp = [int(z) for z in tmp]
        bits = []
        while tmp>0:
            bits.insert(0, tmp%2)
            tmp = int(tmp/2)
        if len(bits)<size:
            bits = [0]*(size-len(bits)) + bits
        output = np.array(bits)
    elif type(tmp) is str:
        tmp = int(tmp,2)
        output = tmp
    elif type(tmp) is np.ndarray or type(tmp) is list:
        #tmp = np.sum(np.flip(tmp) * 2 ** (np.arange(len(tmp))))
        #output = tmp
        output = sum(tmp * 2**(np.arange(len(tmp))[::-1]))
    else:
        print("incorrect input for function binx")
    return output
