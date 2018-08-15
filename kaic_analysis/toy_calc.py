import argparse
import numpy as np
import distutils.dir_util
import pandas as pd
from kaic_analysis.toymodel import Master
import os
import datetime
from multiprocessing import Pool

parser = argparse.ArgumentParser()
parser.add_argument("paramfile", type=str)
parser.add_argument("task_id", type=int)
parser.add_argument("ns", type=int)
args = parser.parse_args()

#folder = 'test'
folder = '/project/biophys/thermodynamics_of_oscillations/toymodel'
distutils.dir_util.mkpath(folder)

paramfilename = folder+'/'+args.paramfile
kwarglist = pd.read_csv(paramfilename)
kwarglist = list(kwarglist.T.to_dict().values())
kwarglist = [kwarglist[args.task_id-1] for k in range(args.ns)]

pool = Pool()
data = pool.map(Master,kwarglist)
pool.close()

t = pd.DataFrame([item['tpass'] for item in data]).T
DelS = pd.DataFrame([item['DelS'] for item in data])
tau = pd.DataFrame([item['tau'] for item in data])

k = 0
for item in data:
    del item['DelS']
    del item['Jmax']
    del item['tau']
    del item['tpass']
    item.update(kwarglist[k])
    k += 1

filenames = [folder+'/'+name+'_'+str(datetime.datetime.now()).split()[0]+'_'+str(args.task_id)+'.csv' for name in ['t','DelS','tau','data']]

t.to_csv(filenames[0],index=False,header=False)
DelS.to_csv(filenames[1],index=False,header=False)
tau.to_csv(filenames[2],index=False,header=False)
pd.DataFrame(data).to_csv(filenames[3])
for name in filenames:
    print('Data saved to ' + name)
