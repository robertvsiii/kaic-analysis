import argparse
import numpy as np
import distutils.dir_util
import pandas as pd
from kaic_analysis.scripts import *
import os
import datetime

parser = argparse.ArgumentParser()
parser.add_argument("task_ID", type=int)
parser.add_argument("sc", type=float)
parser.add_argument("es", type=int)
args = parser.parse_args()

#folder = 'test'
#code_folder = '/users/robertmarsland/Dropbox (Personal)/BU/Thermodynamics of oscillations/KMC_KaiC_rev2'
folder = '/project/biophys/thermodynamics_of_oscillations'
code_folder = '/usr2/postdoc/marsland/KMC_KaiC'
#distutils.dir_util.mkpath(folder)

n_tasks = 20
ATPvec = 1-np.exp(np.linspace(-3,np.log(0.4),n_tasks))
if args.task_ID <= n_tasks:
    param_val = ATPvec[args.task_ID-1]
elif args.task_ID == n_tasks+1:
    param_val = 1-np.exp(-18)

date = str(datetime.datetime.now()).split()[0]

RunExperiment(vol=1,ens_size=args.es,param_val=param_val,param_name='ATPfrac',code_folder=code_folder,
              sample_cnt=int(args.sc),run_number=args.task_ID,use_PCA=True,CIIhyd=True)

tau, DelS, results = ProcessExperiment(run_number=args.task_ID,date=date,
                                       param_name='ATPfrac',param_val=param_val,folder=folder,
                                       code_folder=code_folder,Ncyc=1,all=True)

