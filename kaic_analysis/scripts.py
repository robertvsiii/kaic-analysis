#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Wed Nov 15 10:31:23 2017

@author: robertmarsland
"""

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import subprocess
import os
import pickle
import datetime
from sklearn.decomposition import PCA

StateData = ['ACI', 'ACII', 'CIATP', 'CIIATP', 'pU', 'pT', 'pD', 'pS']

def FormatPath(folder):
    if folder==None:
        folder=''
    else:
        if folder != '':
            if folder[-1] != '/':
                folder = folder+'/'
    return folder

def LoadData(name, folder = None, suffix = '.dat'):
    folder = FormatPath(folder)
    col_ind = list(range(22))
    del col_ind[5]
    return pd.read_table(folder+name+suffix,index_col=0,usecols=col_ind)

def RunModel(paramdict = {}, name = 'data', default = 'default.par', folder = None, extra_mem = False):
    if folder != None:
        cwd = os.getcwd()
        os.chdir(folder)
    linelist = []
    with open(default) as f:
        for line in f:
            for item in paramdict:
                if line[:len(item)] == item:
                    line = item + ' ' + str(paramdict[item]) + '\n'
            if line[:15] == 'output_filename':
                line = 'output_filename ' + name + '\n'
            linelist.append(line)

    with open(name + '.par','w') as f:
        for line in linelist:
            f.write(line)
    if extra_mem:
        subprocess.check_call('ulimit -s 65532; ./KMCKaiC ' + name + '.par', shell = True)
    else:
        subprocess.check_call('./KMCKaiC ' + name + '.par', shell = True)
    if folder != None:
        os.chdir(cwd)
        return LoadData(name, folder=folder)
    else:
        return LoadData(name)

def Current(data,species):
    J = [0]
    t = [data.index[0]]

    center = [np.mean(data[species[0]]),np.mean(data[species[1]])]
    values = [[],[]]

    for k in range(len(data)-1):
        if data[species[0]].iloc[k] < center[0] and data[species[0]].iloc[k+1] < center[0]:
            if data[species[1]].iloc[k] <= center[1] and data[species[1]].iloc[k+1] > center[1]:
                J.append(J[-1]-1)
                t.append(data.index[k])
                values[0].append(data[species[0]].iloc[k])
                values[1].append(data[species[1]].iloc[k])
            if data[species[1]].iloc[k] > center[1] and data[species[1]].iloc[k+1] <= center[1]:
                J.append(J[-1]+1)
                t.append(data.index[k])
                values[0].append(data[species[0]].iloc[k])
                values[1].append(data[species[1]].iloc[k])
    
    J = np.asarray(J,dtype=int)
    t = np.asarray(t,dtype=float)
    T = np.nan
    if len(J) > 1:
        if J[-1]>J[1]:
            T = (t[-1]-t[1])/(J[-1]-J[1])
        
    return t, J, T, center

def Current_PCA(data,center=[1,0]):
    J = [0]
    t = [data.index[0]]
    
    values = [[],[]]
    
    data_PCA = PCA(n_components=2).fit_transform(data[StateData])
    
    for k in range(len(data_PCA)-1):
        if data_PCA[k,1] >= center[1] and data_PCA[k+1,1] >= center[1]:
            if data_PCA[k,0] <= center[0] and data_PCA[k+1,0] > center[0]:
                J.append(J[-1]-1)
                t.append(data.index[k])
                values[0].append(data_PCA[k,0])
                values[1].append(data_PCA[k,1])
            if data_PCA[k,0] > center[0] and data_PCA[k+1,0] <= center[0]:
                J.append(J[-1]+1)
                t.append(data.index[k])
                values[0].append(data_PCA[k,0])
                values[1].append(data_PCA[k,1])
    
    J = np.asarray(J,dtype=int)
    t = np.asarray(t,dtype=float)
    T = np.nan

    if len(J) > 1:
        if J[-1] < 0:
            J = -J
        if J[-1]>J[1]:
            T = (t[-1]-t[1])/(J[-1]-J[1])

    return t, J, T

def EntropyRate(data,name='data',folder=None):
       
    NA = 6.02e23
    conv = 1e-21
    ATPcons_hex = (data['CIATPcons'].iloc[-1] + data['CIIATPcons'].iloc[-1] -
                   data['CIATPcons'].iloc[0] - data['CIIATPcons'].iloc[0])
    ATPcons = (6*conv*NA*FindParam('volume',name,folder=folder)*
               FindParam('KaiC0',name,folder=folder)*ATPcons_hex)
    return (FindParam('Delmu',name,folder=folder)*ATPcons/
            (data.index[-1]-data.index[0]))

def FirstPassageSingleTraj(t,J):
    tau_list = []
    for k in range(2,max(J)+1):
        tau_list.append(t[np.where(J>=k)[0][0]]-t[np.where(J>=k-1)[0][0]])
    return tau_list

def FindParam(param,par_file,folder=None):
    folder = FormatPath(folder)
                
    if param == 'Delmu':
        paramdict = {}
        with open(folder+par_file+'.par') as f:
            for line in f:
                words = line.split()
                if words != []:
                    if words[0] in ['Khyd','ATPfrac','Piconc']:
                        paramdict[words[0]] = float(words[1])
                
        return np.log(paramdict['Khyd']/paramdict['Piconc']) + np.log(1/((1/paramdict['ATPfrac'])-1))
    else:
        with open(folder+par_file+'.par') as f:
            for line in f:
                words = line.split()
                if words != []:
                    if words[0] == param:
                        return float(words[1])

def EntropyProduction(data,name='data'):
    NA = 6.02e23
    conv = 1e-21
    ATPcons = 6*conv*NA*FindParam('volume',name)*FindParam('KaiC0',name)*(data['CIATPcons'] + data['CIIATPcons'])
    return FindParam('Delmu',name)*ATPcons

def Ensemble(paramdict,ns,species=['pT','pS'],folder=None,run_number=1):
    results = []
    Tvec = []
    Sdotvec = []
    
    path = FormatPath(folder)
    
    date = str(datetime.datetime.now()).split()[0]
    name = '_'.join([str(run_number),date])
    filename = path + 'RawData_' + name + '.dat'

    for k in range(ns):
        paramdict['rnd_seed'] = np.random.rand()*1000000
        data = None
        count = 0
        while data is None and count < 10:
            try:
                datname = 'data_'+str(np.random.randint(1000000))
                data = RunModel(paramdict=paramdict,name=datname,folder=folder)
            except:
                subprocess.check_call('rm -f '+path+datname+'.par', shell = True)
                count += 1
        assert data is not None, 'KMCKaiC failed to run.'

        t, J, T_new, center = Current(data,species)
        Sdot_new = EntropyRate(data,name=datname,folder=folder)
        Tvec.append(T_new)
        Sdotvec.append(Sdot_new)
        results.append({'t': t, 'J': J})
        subprocess.check_call('rm -f '+'\''+path+datname+'.dat'+'\'', shell = True)
        subprocess.check_call('rm -f '+'\''+path+datname+'.par'+'\'', shell = True)
        T = np.nanmean(Tvec)
        Sdot = np.nanmean(Sdotvec)
        with open(filename,'wb') as f:
            pickle.dump([results,T,Sdot],f)

def Ensemble_PCA(paramdict,ns,folder=None,run_number=1):
    results = []
    Tvec = []
    Sdotvec = []
    
    path = FormatPath(folder)
    
    date = str(datetime.datetime.now()).split()[0]
    name = '_'.join([str(run_number),date])
    filename = path + 'RawData_' + name + '.dat'
    
    for k in range(ns):
        paramdict['rnd_seed'] = np.random.rand()*1000000
        data = None
        count = 0
        while data is None and count < 10:
            try:
                datname = 'data_'+str(np.random.randint(1000000))
                data = RunModel(paramdict=paramdict,name=datname,folder=folder)
            except:
                subprocess.check_call('rm -f '+path+datname+'.par', shell = True)
                count += 1
        assert data is not None, 'KMCKaiC failed to run.'
        
        t, J, T_new = Current_PCA(data)
        Sdot_new = EntropyRate(data,name=datname,folder=folder)
        Tvec.append(T_new)
        Sdotvec.append(Sdot_new)
        results.append({'t': t, 'J': J})
        subprocess.check_call('rm -f '+'\''+path+datname+'.dat'+'\'', shell = True)
        subprocess.check_call('rm -f '+'\''+path+datname+'.par'+'\'', shell = True)
        T = np.nanmean(Tvec)
        Sdot = np.nanmean(Sdotvec)
        with open(filename,'wb') as f:
            pickle.dump([results,T,Sdot],f)

def FirstPassage(results,Ncyc = 1,all=False):
    tau = []
    if all:
        for item in results:
            tau = tau + FirstPassageSingleTraj(item['t'],item['J'])
    else:
        for item in results:
            inds1 = np.where(item['J'] >= 1)[0]
            inds2 = np.where(item['J'] >= 1+Ncyc)[0]
            if len(inds1) != 0 and len(inds2) != 0:
                t1 = item['t'][inds1[0]]
                t2 = item['t'][inds2[0]]
                tau.append(t2-t1)
            else:
                tau.append(np.nan)
        
    return tau

def LoadExperiment(param_name,run_numbers,date,folder='data'):
    folder = FormatPath(folder)
    
    name = '_'.join([param_name,str(run_numbers[0]),date])
    filename1 = folder + 'FirstPassageData_' + name + '.csv'
    filename2 = folder + 'Sdot_' + name + '.csv'
    filename3 = folder + 'AllData_' + name + '.dat'
    
    tau=pd.read_csv(filename1,index_col=0)
    Sdot=pd.read_csv(filename2,index_col=0)
    with open(filename3,'rb') as f:
        results=pickle.load(f)
       
    for run_number in run_numbers[1:]:
        name = '_'.join([param_name,str(run_number),date])
        filename1 = folder + 'FirstPassageData_' + name + '.csv'
        filename2 = folder + 'Sdot_' + name + '.csv'
        filename3 = folder + 'AllData_' + name + '.dat'
    
        tau = tau.join(pd.read_csv(filename1,index_col=0))
        Sdot = Sdot.join(pd.read_csv(filename2,index_col=0))
        with open(filename3,'rb') as f:
            results_new=pickle.load(f)
        results.update(results_new)
        
    return tau, Sdot, results

def RunExperiment(vol = 0.5, param_val = 25, param_name = 'Delmu', ens_size = 5, CIIhyd = True,
                  sample_cnt = 3e6, code_folder = None, run_number = 1, use_PCA = False):
    
    paramdict = {}
    paramdict['volume'] = vol
    paramdict['sample_cnt'] = sample_cnt
    paramdict['tequ'] = 50

    if not CIIhyd:
        paramdict['kCIIhyd0'] = 0.1
    
    if param_name == 'Delmu':
        paramdict['Khyd'] = (np.exp(param_val)*FindParam('Piconc','default',folder=code_folder)*
                             ((1/FindParam('ATPfrac','default',folder=code_folder))-1))
    else:
        paramdict[param_name] = param_val

    if use_PCA:
        Ensemble_PCA(paramdict,ens_size,folder=code_folder,run_number=run_number)
    else:
        Ensemble(paramdict,ens_size,folder=code_folder,run_number=run_number)
    
    
def ProcessExperiment(run_number = 1, date = str(datetime.datetime.now()).split()[0], all = False,
                      param_name = 'Delmu', param_val = 20, folder = 'data', code_folder = None, Ncyc = 30):
    
    if all:
        Ncyc = 1
    
    folder = FormatPath(folder)
    code_folder = FormatPath(code_folder)
    
    filename0 = code_folder + 'RawData_' + '_'.join([str(run_number),date]) + '.dat'
    
    name = '_'.join([param_name,str(run_number),date])
    filename1 = folder + 'FirstPassageData_' + name + '.csv'
    filename2 = folder + 'Sdot_' + name + '.csv'
    filename3 = folder + 'AllData_' + name + '.dat'
    
    keyname = param_name + ' = ' + str(param_val)
    
    results = {}
    tau = {}
    Sdot = {}
    
    with open(filename0,'rb') as f:
        results[keyname], T, Sdot[keyname] = pickle.load(f)
    tau[keyname] = FirstPassage(results[keyname],Ncyc=Ncyc,all=all)
        
    tau = pd.DataFrame.from_dict(tau)
    tau.to_csv(filename1)
    Sdot = pd.DataFrame.from_dict(Sdot,orient='index').T
    Sdot.to_csv(filename2)
    with open(filename3,'wb') as f:
        pickle.dump(results,f)
        
    return tau, Sdot, results

def PlotExperiment(ex_out,tmax = 3000., taumax = 3000., nbins = 50):
    tau = ex_out[0]
    Sdot = ex_out[1]
    results = ex_out[2]
    ns2 = len(tau.keys())
    tbins = np.linspace(0,taumax,nbins)
    fig, axs = plt.subplots(ns2,2,sharex='col',figsize = (8,10))
    
    paramlist = []
    for name in tau.keys():
        paramlist.append(float(name.split()[-1]))
    param_name = tau.keys()[0].split()[0]
    paramlist.sort()


    k = 0
    eps = []
    Sdotmean = []
    T = []
    for paramval in paramlist:
        name = param_name + ' = ' + str(paramval)
        for item in results[name]:
            if type(item['t']) == list or type(item['t']) == np.ndarray:
                if len(item['t']) > 1:
                    axs[k,0].plot(item['t']-item['t'][1],item['J']-item['J'][1])
        tau[name].hist(ax=axs[k,1],bins=tbins)
        axs[k,1].set_yticks(())
        eps.append(tau[name].var()/tau[name].mean()**2)
        T.append(tau[name].mean())
        Sdotmean.append(Sdot[name])
        k += 1

    axs[int(round(ns2*1./2)),0].set_ylabel('Number of Cycles')
    axs[-1,0].set_xlabel('Time (hrs)')
    axs[-1,1].set_xlabel(r'$\tau$ (hrs)')
    axs[-1,0].set_xlim((0,tmax))
    axs[-1,1].set_xlim((0,taumax))
    
    plt.show()
        
    DelSmean = np.asarray(Sdotmean)*np.asarray(T)
    eps = np.asarray(eps)
    DelSrange = np.linspace(min(DelSmean)*0.75,max(DelSmean)*1.25,100)
    fig2, ax2 = plt.subplots(1)
    ax2.semilogy(DelSmean,eps,'o-')
    ax2.semilogy(DelSrange,2./DelSrange,'k',linewidth = 3)
    ax2.set_xlabel(r'$\langle \Delta S_c \rangle$')
    ax2.set_ylabel(r'var($\tau$)/$\langle \tau\rangle^2$')

    plt.show()
