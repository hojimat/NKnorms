import numpy as np
from itertools import combinations as comb

def binary_combinations(N,R):
    tmp = []
    idx = comb(range(N),R)
    for i in idx:
        A = [0]*N
        for j in i:
            A[j] = 1
        tmp.append(A)
    output = np.reshape(tmp,(-1,N))
    return(output)

def interaction_matrix(N,R,diag=None):
    if N==R:
        tmp = np.ones((N,R),dtype=int)
        print(tmp)
        return tmp
    elif N<R:
        print("Incorrect interaction matrix. Check specification.")
        return
    tmp = np.zeros(2*N,dtype=int).reshape(2,N)
    #tmp = np.eye(N,dtype=int)
    cl = binary_combinations(N,R)
    for i in range(N):
        colsums = np.sum(tmp,0)
        # filter excess ones
        idx = np.empty(0,dtype=int)
        Rx = np.repeat(R,N)
        #if diag is not None:
        #    Rx[:i]-=diag
        for j in np.where(colsums == R)[0]:
            k = np.where(cl[:,j]>0)[0]
            idx = np.union1d(idx,k)
        cl = np.delete(cl,idx,0)
        # filter excess zeros
        inx = np.empty(0,dtype=int)
        for j in np.where(colsums+N-i == R)[0]:
            k = np.where(cl[:,j]==0)[0]
            inx = np.union1d(inx,k)
        cl = np.delete(cl,inx,0)

        cl = cl.copy()
        if diag is not None:
            ivx = np.where(cl[:,i]==diag)[0]
            cli = cl[ivx,]
            clk = (cli + colsums)[:,i+1:]
            #ikx = np.where(clk==R)[0]
            tp = N-i-1 if diag==0 else 0 # tuning parameter
            ikx = np.where(clk+tp==R)[0]
            cli = np.delete(cli,ikx,0)


        ncl = cl.shape[0]
        ncli = cli.shape[0]
        if ncli > 0:
            ind = np.random.choice(ncli)
            tmp = np.vstack((tmp,cli[ind,:]))
            #print(tmp[2:,:])
        elif ncli==0 and ncl>0:
            print('Error creating non-zero diagonals. Rerun the function')
            return 0
        else:
            print('Incorrect interaction matrix. Check the dimensions.')
            return 
    output = np.delete(tmp,[0,1],0)
    print(output)
    #print(output.sum(0))
    return(output)


mc = 0
while mc <1:
    interaction_matrix(16,11,1)
    mc +=1
   # maybe for the time being reiterate under errors; think more;





#def filter_set(vec,cond):
#   tmp = [z for (z,v) in zip(vec,cond) if v]
#   output = tmp
#   return output
#
#def xinteraction_matrix(N,R,eye=False):
#    output = None
#    if eye:
#        output = np.eye(N,dtype=int)
#    else:
#        output = np.zeros((N,N),dtype=int)
#
#    for i in range(N):
#        # A. Determine indices
#        box = []
#        a = list(range(N))
#        colsum = output.sum(0)
#
#        # 1. (re)add diagonal
#        if eye:
#            box.append(i)
#
#        # 2. add criticals
#        cond1 = colsum+N-i == R
#        tmp = filter_set(a,cond1)
#        if eye:
#            tmp = [z for z in tmp if z!=i]
#        box += tmp
#
#        # 3. randomize the rest
#        if len(box) < R:
#            cond2 = colsum <= R
#            tmp = filter_set(a,cond2)
#            tmp = [z for z in tmp if z not in box]
#            tmp = [z for z in tmp if colsum[z]<=R]
#            tmp = np.random.choice(tmp,(R-len(box)),False)
#            tmp = list(tmp)
#            box += tmp
#
#        # B. Set values
#        for j in box:
#            output[i,j] = 1
#
#    print(output)
#    print(output.sum(0))
#    return output
#
#def ointeraction_matrix(N,R,eye=False):
#    if N==R:
#        tmp = np.ones((N,R),dtype=int)
#        print(tmp)
#        return tmp
#    elif N<R:
#        print("Incorrect interaction matrix. Check specification.")
#        return
#    tmp = np.zeros(2*N,dtype=int).reshape(2,N)
#    cl = binary_combinations(N,R)
#    for i in range(N):
#        colsums = np.sum(tmp,0)
#
#        idx = np.empty(0,dtype=int)
#        for j in np.where(colsums>=R)[0]:
#            k = np.where(cl[:,j]>0)[0]
#            idx = np.union1d(idx,k)
#        cl = np.delete(cl,idx,0)
#
#        inx = np.empty(0,dtype=int)
#        for j in np.where(colsums+N-i == R)[0]:
#            k = np.where(cl[:,j]==0)[0]
#            inx = np.union1d(inx,k)
#        cl = np.delete(cl,inx,0)
#
#        ncl = cl.shape[0]
#        if ncl > 0:
#            ind = np.random.choice(ncl)
#            tmp = np.vstack((tmp,cl[ind,:]))
#        else:
#            print('Incorrect interaction matrix. Check the dimensions.')
#    output = np.delete(tmp,[0,1],0)
#    print(output)
#    #print(output.sum(0))
#    return(output)
#
#    # remove first row
#
#
#def MAINinteraction_matrix(N,R,eye=False):
#    if N==R:
#        tmp = np.ones((N,R),dtype=int)
#        print(tmp)
#        return tmp
#    elif N<R:
#        print("Incorrect interaction matrix. Check specification.")
#        return
#    tmp = np.zeros(2*N,dtype=int).reshape(2,N)
#    cl = binary_combinations(N,R)
#    for i in range(N):
#        colsums = np.sum(tmp,0)
#
#        idx = np.empty(0,dtype=int)
#        for j in np.where(colsums>=R)[0]:
#            k = np.where(cl[:,j]>0)[0]
#            idx = np.union1d(idx,k)
#        cl = np.delete(cl,idx,0)
#
#        inx = np.empty(0,dtype=int)
#        for j in np.where(colsums+N-i == R)[0]:
#            k = np.where(cl[:,j]==0)[0]
#            inx = np.union1d(inx,k)
#        cl = np.delete(cl,inx,0)
#
#        cli = cl.copy()
#        if eye:
#            ivx = np.where(cl[:,i]>0)[0]
#            cli = cl[ivx,]
#
#        ncl = cl.shape[0]
#        ncli = cli.shape[0]
#        if ncli > 0:
#            ind = np.random.choice(ncli)
#            tmp = np.vstack((tmp,cli[ind,:]))
#        elif ncli==0 and ncl>0:
#            print('Error creating non-zero diagonals. Rerun the function')
#            return 0
#        else:
#            print('Incorrect interaction matrix. Check the dimensions.')
#            return 
#    output = np.delete(tmp,[0,1],0)
#    print(output)
#    #print(output.sum(0))
#    return(output)
