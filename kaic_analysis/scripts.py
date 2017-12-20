#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Wed Nov 15 10:31:23 2017

@author: robertmarsland
"""

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import subprocess
import os
import pickle
import datetime

StateData = ['phos_frac', 'Afree', 'ACI', 'ACII', 'CIATP', 'CIIATP', 'Ttot', 'Stot', 'pU', 'pT', 'pD', 'pS']

def LoadData(name, folder = '', suffix = '.dat'):
    col_ind = range(22)
    #del col_ind[5]
    data = pd.read_table(folder+name+suffix,index_col=0,usecols=col_ind)
    return data.loc[(data!=0).any(1)]

def RunModel(paramdict = {}, name = 'data', default = 'default.par', folder = ''):
    linelist = []
    with open(folder + default) as f:
        for line in f:
            for item in paramdict:
                if line[:len(item)] == item:
                    line = item + ' ' + str(paramdict[item]) + '\n'
            if line[:15] == 'output_filename':
                line = 'output_filename ' + name + '\n'
            linelist.append(line)

    with open(folder + name + '.par','w') as f:
        for line in linelist:
            f.write(line)
    subprocess.check_call(folder+'KMCKaiC ' + name + '.par', shell = True)
    
    return LoadData(name, folder=folder)

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
    t = np.asarray(t)
    if len(J) > 0:
        T = (t[-1]-t[1])/(J[-1]-J[1])
    else:
        T = np.nan
    return t, J, T, center

def EntropyRate(data,name='data'):
    NA = 6.02e23
    conv = 1e-21
    ATPcons_hex = (data['CIATPcons'].iloc[-1] + data['CIIATPcons'].iloc[-1] -
                   data['CIATPcons'].iloc[0] - data['CIIATPcons'].iloc[0])
    ATPcons = 6*conv*NA*FindParam('volume',name)*FindParam('KaiC0',name)*ATPcons_hex
    return FindParam('Delmu',name)*ATPcons/(FindParam('tend',name)-FindParam('tequ',name))

def FirstPassageSingleTraj(t,J):
    tau_list = []
    for k in range(2,max(J)+1):
        tau_list.append(t[np.where(J>=k)[0][0]]-t[np.where(J>=k-1)[0][0]])
    return np.asarray(tau_list)

def FindParam(param,par_file,folder = ''):
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

def Ensemble(paramdict,ns,species=['pT','pS'],folder='',savename = 'data_processed',datname = 'data'):
    results = []
    for k in range(ns):
        paramdict['rnd_seed'] = np.random.rand()*100
        data = None
        while data is None:
            try:
                data = RunModel(paramdict=paramdict,name = datname,folder=folder)
            except:
                pass
        t, J, T, center = Current(data,species)
        Sdot = EntropyRate(data,name = datname)
        results.append({'t': t, 'J': J, 'T': T, 'DelS': Sdot*T})
        
    return results

def FirstPassage(results,Ncyc = 1):
    tau = []
    DelS = []
    for item in results:
        inds1 = np.where(item['J'] >= 1)[0]
        inds2 = np.where(item['J'] >= 1+Ncyc)[0]
        if len(inds1) != 0 and len(inds2) != 0:
            t1 = item['t'][inds1[0]]
            t2 = item['t'][inds2[0]]
            tau.append(t2-t1)
        else:
            tau.append(np.nan)
        DelS.append(item['DelS'])
        
    return tau, DelS

def Experiment(vol = 0.5, ATPmin = 0.5, ATPmax = 0.999, ns1 = 10, ns2 = 5, 
               paramdict = {}, folder = '', Ncyc = 50, tend = 3e3,
               code_folder = './'):
    if ATPmax == 1:
        return 'ATPfrac must be less than 1 to get finite entropy production.'
    
    filename1 = folder + 'FirstPassageData_vol_' + str(vol) + '_' + str(datetime.datetime.now()).split()[0] + '.csv'
    filename2 = folder + 'DelS_vol_' + str(vol) + '_' + str(datetime.datetime.now()).split()[0] + '.csv'
    filename3 = folder + 'AllData_vol_' + str(vol) + '_' + str(datetime.datetime.now()).split()[0] + '.dat'
    
    paramdict['volume'] = vol
    paramdict['tend'] = tend
    paramdict['tequ'] = 50
    results = {}
    tau = {}
    DelS = {}
    
    for ATPfrac in np.linspace(ATPmin,ATPmax,ns2):
        keyname = 'ATPfrac = '+str(ATPfrac)
        paramdict['ATPfrac'] = ATPfrac
        results[keyname] = Ensemble(paramdict,ns1,folder=code_folder)
        tau[keyname], DelS[keyname] = FirstPassage(results[keyname],Ncyc=Ncyc)
        
    tau = pd.DataFrame.from_dict(tau)
    tau.to_csv(filename1)
    DelS = pd.DataFrame.from_dict(DelS)
    DelS.to_csv(filename2)
    with open(filename3,'wb') as f:
        pickle.dump(results,f)
        
    return tau, DelS, results

def PlotExperiment(ex_out,tmax = 3000., taumax = 3000., DelSmax = 6000000., nbins = 50):
    tau = ex_out[0]
    DelS = ex_out[1]
    results = ex_out[2]
    ns2 = len(tau.keys())
    tbins = np.linspace(0,taumax,nbins)
    sbins = np.linspace(0,DelSmax,nbins)
    fig, axs = plt.subplots(ns2,3,sharex='col',figsize = (8,10))
    
    ATPfraclist = []
    for name in tau.keys():
        ATPfraclist.append(float(name.split()[-1]))
    ATPfraclist.sort()


    k = 0
    eps = []
    DelSmean = []
    for ATPfrac in ATPfraclist:
        name = 'ATPfrac = '+str(ATPfrac)
        for item in results[name]:
            if type(item['t']) == list or type(item['t']) == np.ndarray:
                if len(item['t']) > 1:
                    axs[k,0].plot(item['t']-item['t'][1],item['J']-item['J'][1])
        tau[name].hist(ax=axs[k,1],bins=tbins)
        axs[k,1].set_yticks(())
        DelS[name].hist(ax=axs[k,2],bins=sbins)
        axs[k,2].set_yticks(())
        eps.append(tau[name].var()/tau[name].mean()**2)
        DelSmean.append(DelS[name].mean())
        k += 1

    axs[int(round(ns2*1./2)),0].set_ylabel('Number of Cycles')
    axs[-1,0].set_xlabel('Time (hrs)')
    axs[-1,1].set_xlabel(r'$\tau$ (hrs)')
    axs[-1,2].set_xlabel(r'$\Delta S_c\,(k_B)$')
    axs[-1,0].set_xlim((0,tmax))
    axs[-1,1].set_xlim((0,taumax))
    
    plt.show()
        
    DelSmean = np.asarray(DelSmean)
    eps = np.asarray(eps)
    DelSrange = np.linspace(min(DelSmean)*0.75,max(DelSmean)*1.25,100)
    fig2, ax2 = plt.subplots(1)
    ax2.semilogy(DelSmean,eps,'o-')
    ax2.semilogy(DelSrange,2./DelSrange,'k',linewidth = 3)
    ax2.set_xlabel(r'$\langle \Delta S_c \rangle$')
    ax2.set_ylabel(r'var($\tau$)/$\langle \tau\rangle^2$')

    plt.show()
