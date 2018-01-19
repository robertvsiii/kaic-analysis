import argparse
import numpy as np
import distutils.dir_util
import pandas as pd
from kaic_analysis.scripts import *
import os
import datetime

parser = argparse.ArgumentParser()
parser.add_argument("task_ID", type=int)
parser.add_argument("param", type=str)
parser.add_argument("scale", type=float)
parser.add_argument("sc", type=float)
parser.add_argument("es", type=int)
args = parser.parse_args()

#folder = 'test'
#code_folder = '/users/robertmarsland/Dropbox (Personal)/BU/Thermodynamics of oscillations/KMC_KaiC_rev2'
folder = '/project/biophys/thermodynamics_of_oscillations'
code_folder = '/usr2/postdoc/marsland/KMC_KaiC'
#distutils.dir_util.mkpath(folder)

param_val = args.task_ID*args.scale

RunExperiment(ens_size=args.es,param_val=param_val,param_name=args.param,code_folder=code_folder,
              sample_cnt=int(args.sc),run_number=args.task_ID)

tau, DelS, results = ProcessExperiment(run_number=args.task_ID,date=str(datetime.datetime.now()).split()[0], 
                                       param_name=args.param,param_val=param_val,folder=folder,
                                       code_folder=code_folder,Ncyc=30)
