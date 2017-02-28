import numpy as np
import math as ma

def IALM(D,mu,rho):
    # incomplete alternating Lagrangian
    # method (IALM) for solving the
    # matrix completion problem

    # thresholds
    ep1 = 1.e-7
    ep2 = 1.e-6
    Dn  = np.linalg.norm(D,'fro')

    # projector matrix
    PP = (D == 0)
    P  = PP.astype(np.float)

    # initialization
    m,n  = np.shape(D)
    Y    = np.zeros((m,n))
    Eold = np.zeros((m,n))

    # iteration
    for i in range(1,1000):

        # compute SVD
        tmp   = D - Eold + Y/mu
        U,S,V = np.linalg.svd(tmp,full_matrices=False)

        # threshold and patch matrix back together
        ss = S-(1/mu)
        s2 = np.clip(ss,0,max(ss))
        A  = np.dot(U,np.dot(np.diag(s2),V))

        # project
        Enew = P*(D - A + Y/mu)
        DAE  = D - A - Enew
        Y   += mu*DAE

        # check residual and (maybe) exit
        r1   = np.linalg.norm(DAE,'fro')
        resi = r1/Dn
        print i,' residual ',resi
        if (resi < ep1):
            break

        # adjust mu-factor
        muf  = np.linalg.norm((Enew-Eold),'fro')
        fac  = min(mu,ma.sqrt(mu))*(muf/Dn)
        if (fac < ep2):
            mu *= rho

        # update E and go back
        Eold = np.copy(Enew)

    E = np.copy(Enew)
    return A,E


if __name__ == '__main__':

    # matrix completion problem (Netflix problem)

    # size of problem
    m = 500                     # rows
    n = 150                     # columns
    r = int(round(min(m,n)/3))  # rank

    # construct low-rank matrix
    U = np.random.random((m,r))
    V = np.random.random((r,n))
    A = np.dot(U,V)

    # sampling matrix
    PP = (np.random.random((m,n)) > 0.433)
    P  = PP.astype(np.float)
    # number of non-zero elements
    Omega = np.count_nonzero(P)

    # data matrix
    D = P*A
    fratio = float(Omega)/(m*n)
    print 'fill ratio ', fratio

    # initialize parameters
    mu  = 1./np.linalg.norm(D,2)
    rho = 1.2172 + 1.8588*fratio

    # call IALM-algorithm
    AA,EE = IALM(D,mu,rho)

    # compare

    print('\n')
    print('Data matrix')
    print D[0:5,0:5]
    print('\n')
    print('Recovered matrix')
    print AA[0:5,0:5]
    print('\n')
    print('Original matrix')
    print A[0:5,0:5]
