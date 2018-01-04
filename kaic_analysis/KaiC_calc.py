import argparse
import numpy as np
import distutils.dir_util
import pandas as pd
from kaic_analysis.scripts import *
import os

parser = argparse.ArgumentParser()
parser.add_argument("task_ID", type=int)
parser.add_argument("param", type=str)
parser.add_argument("scale", type=float)
parser.add_argument("es", type=int)
args = parser.parse_args()

#folder = 'test'
#code_folder = '/users/robertmarsland/Dropbox (Personal)/BU/Thermodynamics of oscillations/KMC_KaiC_rev2/'
folder = '/project/biophys/thermodynamics_of_oscillations/'
code_folder = '/usr2/postdoc/marsland/KMC_KaiC/'
#distutils.dir_util.mkpath(folder)

sc = 1e7
param_val = args.task_ID*args.scale

tau, DelS, results = Experiment(ens_size=args.es,param_min=param_val,param_max=args.max,
                                param_name=args.param,folder=folder,code_folder=code_folder,sample_cnt=sc)