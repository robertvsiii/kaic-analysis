import argparse
import numpy as np
import distutils.dir_util
import pandas as pd
from kaic_analysis.scripts import *
import os
import datetime

parser = argparse.ArgumentParser()
parser.add_argument("task_ID", type=int)
parser.add_argument("date", type=str)
args = parser.parse_args()

#folder = 'test'
#code_folder = '/users/robertmarsland/Dropbox (Personal)/BU/Thermodynamics of oscillations/KMC_KaiC_rev2'
folder = '/project/biophys/thermodynamics_of_oscillations'
code_folder = '/usr2/postdoc/marsland/KMC_KaiC'
#distutils.dir_util.mkpath(folder)

ATPvec = 1-np.exp(np.linspace(-6,np.log(0.4),20))
param_val = ATPvec[args.task_ID-1]

tau, DelS, results = ProcessExperiment(run_number=args.task_ID,date=args.date,
                                       param_name='ATPfrac',param_val=param_val,folder=folder,
                                       code_folder=code_folder,Ncyc=1,all=True)

