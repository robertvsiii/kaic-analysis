import matplotlib.pyplot as plt
import pandas as pd
from sklearn.decomposition import PCA
import numpy as np
from numpy.random import RandomState
from multiprocessing import current_process
from datetime import date, datetime

def ComputeRates(f,du,A,N,M):
    dU = du.dot(f)
    kp = N*np.exp(dU)
    km =N*np.exp(dU-A/N)
    fwd_prop = kp*f*M
    rev_prop = km*np.roll(f,-1)*M
    prop = [rev_prop,fwd_prop]
    return prop

def UpdateState(f,direction,site,N,M):
    f[site] = f[site] - 2*(direction-0.5)/M
    f[np.mod(site+1,N)] = f[np.mod(site+1,N)] + 2*(direction-0.5)/M
    f[f<0] = 0
    return f

def CountCurrent(t,f,f_old,projection,J,Jmax,Jmin,tpass_f,tpass_r):
    tau_f = np.nan
    tau_r = np.nan
    f_old_PCA = projection.transform([f_old])[0]
    f_PCA = projection.transform([f])[0]
    if f_old_PCA[1] >= 0 and f_PCA[1] >= 0:
        if f_old_PCA[0] <= 0 and f_PCA[0] > 0:
            J += -1
        if f_old_PCA[0] > 0 and f_PCA[0] <= 0:
            J += 1
        if J > Jmax:
            Jmax = J
            tau_f = t-tpass_f
            tpass_f = t
        if J < Jmin:
            Jmin = J
            tau_r = t-tpass_r
            tpass_r = t
    return J, Jmax, Jmin, tpass_f, tpass_r, tau_f, tau_r


def SimulateClockKinetic(teq=1,Ncyc=30,tmax=np.inf,nsteps=1000,N=4,M=10,A=1,C=10,projection=None):
    
    worker = str(current_process())
    try:
        workernum = worker.split('-')[1].split(',')[0]
    except:
        workernum = str(1)

    rst = RandomState(int(workernum)*int(str(datetime.now())[-3:]))

    #INITIALIZE
    N = int(N)
    M = int(M)
    nsteps = int(nsteps)
    
    du = C*np.asarray([np.sin((np.arange(N)-m)*2*np.pi/N) for m in range(N)])
    
    f = np.ones(N)/N
    t = 0
    
    DelS = 0
    J = 0
    Jmax = 0
    Jmin = 0
    tpass_f = np.nan
    tpass_r = np.nan
    tpassvec_f = []
    tpassvec_r = []
    tauvec_f = []
    tauvec_r = []
    
    ftraj = []
    tvec = []
    DelSvec = []
    Jmaxvec = []
    Jminvec = []
    Jvec = []
    
    #Initial Rates
    prop = ComputeRates(f,du,A,N,M)
    while max([Jmax,-Jmin]) <= Ncyc and t < tmax:
        noise = rst.rand(nsteps,3)
        for m in range(nsteps):
            #CHOOSE TIME OF NEXT EVENT
            ktot = np.sum(prop[0]+prop[1])
            dt = -(1/ktot)*np.log(noise[m,0])
            t += dt
            
            #CHOOSE DIRECTION OF NEXT EVENT
            direction = int(noise[m,1] > np.sum(prop[0])/ktot)
            #CHOOSE REACTION FOR NEXT EVENT
            p = prop[direction]/np.sum(prop[direction])
            site = np.argmax(noise[m,2] <= np.cumsum(p))
            
            #UPDATE STATE
            f_old = f.copy()
            f = UpdateState(f,direction,site,N,M)
            
            #COMPUTE RATES
            prop_old = prop.copy()
            prop = ComputeRates(f,du,A,N,M)
            
            if t > teq:
                #UPDATE ENTROPY CHANGE
                DelS += np.log(prop_old[direction][site]/prop[1-direction][site])
                
                #UPDATE CURRENT
                if projection is not None:
                    J, Jmax, Jmin, tpass_f, tpass_r, tau_f, tau_r = CountCurrent(t,f,f_old,projection,J,Jmax,Jmin,tpass_f,tpass_r)
                    if np.isfinite(tau_f):
                        tauvec_f.append(tau_f)
                        Jmaxvec.append(Jmax)
                        tpassvec_f.append(tpass_f)
                    if np.isfinite(tau_r):
                        tauvec_r.append(tau_r)
                        Jminvec.append(Jmin)
                        tpassvec_r.append(tpass_r)

    
        if t > teq:
            #RECORD OUTPUT
            ftraj.append(f.copy())
            tvec.append(t)
            DelSvec.append(DelS)
            Jvec.append(J)

    tvec = np.asarray(tvec)
    ftraj = np.asarray(ftraj)
    DelSvec = np.asarray(DelSvec)
    
    if len(tauvec_r) > len(tauvec_f):
        Jmaxvec = -np.asarray(Jminvec)
        Jvec = -np.asarray(Jvec)
        tauvec = np.asarray(tauvec_r)
        tpassvec = np.asarray(tpassvec_r)
    else:
        Jmaxvec = np.asarray(Jmaxvec)
        Jvec = np.asarray(Jvec)
        tauvec = np.asarray(tauvec_f)
        tpassvec = np.asarray(tpassvec_f)

    if projection is not None:
        T = np.mean(tauvec)
        D = np.var(tauvec)/T
        Sdot = DelS/t

    if projection is None:
        return {'t':tvec, 'f':ftraj}
    else:
        return {'DelS':DelSvec, 'tpass':tpassvec, 'Jmax':Jmaxvec,
            'tau':tauvec, 'D': D, 'T': T, 'Sdot': Sdot}

def Master(kwargs_in):
    kwargs = kwargs_in.copy()
    nsteps = kwargs['nsteps']
    kwargs['tmax'] = kwargs['t_init']
    kwargs['nsteps'] = kwargs['n_init']
    del kwargs['t_init']
    del kwargs['n_init']
    out = SimulateClockKinetic(**kwargs)
    model = PCA(n_components=2).fit(out['f'])
    del kwargs['tmax']
    kwargs.update({'nsteps':nsteps,'projection':model})
    out = SimulateClockKinetic(**kwargs)
    print('Finished Job')
    
    return out
