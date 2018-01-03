import argparse
import numpy as np
import distutils.dir_util
import pandas as pd
from kaic_analysis.scripts import *
import os

parser = argparse.ArgumentParser()
parser.add_argument("param", type=str)
parser.add_argument("min", type=float)
parser.add_argument("max", type=float)
parser.add_argument("ns", type=int)
parser.add_argument("es", type=int)
args = parser.parse_args()

#folder = 'test'
#code_folder = '/users/robertmarsland/Dropbox (Personal)/BU/Thermodynamics of oscillations/KMC_KaiC_rev2/'
folder = '/project/biophys/thermodynamics_of_oscillations/'
code_folder = '/usr2/postdoc/marsland/KMC_KaiC/'
#distutils.dir_util.mkpath(folder)

sc = 3e6

tau, DelS, results = Experiment(vol=1,n_steps=args.ns,ens_size=args.es,param_min=args.min,param_max=args.max,
                                param_name=args.param,folder=folder,code_folder=code_folder,sample_cnt=sc)