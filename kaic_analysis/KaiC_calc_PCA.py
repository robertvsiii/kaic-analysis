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

ATPvec = 1-np.exp(np.linspace(np.log(1e-6),np.log(0.4),20))
param_val = ATPvec[args.task_ID-1]

RunExperiment(ens_size=args.es,param_val=param_val,param_name='ATPfrac',code_folder=code_folder,
              sample_cnt=int(args.sc),run_number=args.task_ID,use_PCA=True)

tau, DelS, results = ProcessExperiment(run_number=args.task_ID,date=str(datetime.datetime.now()).split()[0],
                                       param_name='ATPfrac',param_val=param_val,folder=folder,
                                       code_folder=code_folder,Ncyc=1,all=True)

