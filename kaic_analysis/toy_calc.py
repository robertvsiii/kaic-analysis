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
args = parser.parse_args()

#folder = 'test'
folder = '/project/biophys/thermodynamics_of_oscillations/toymodel'
distutils.dir_util.mkpath(folder)

paramfilename = folder+'/'+args.paramfile
kwarglist = pd.read_csv(paramfilename)
kwarglist = list(kwarglist.T.to_dict().values())

pool = Pool()
data = pool.map(Master,kwarglist)
pool.close()

f = pd.DataFrame([item['f'][:,0] for item in data])
f1 = pd.DataFrame([item['f1'] for item in data])
f2 = pd.DataFrame([item['f2'] for item in data])
t = pd.DataFrame([item['t'] for item in data])
Jmax = pd.DataFrame([item['Jmax'] for item in data])
DelS = pd.DataFrame([item['DelS'] for item in data])
tau = pd.DataFrame([item['tau'] for item in data])

k = 0
for item in data:
    del item['f']
    del item['t']
    del item['f1']
    del item['f2']
    del item['DelS']
    del item['Jmax']
    del item['tau']
    item.update(kwarglist[k])
    k += 1

filenames = [folder+'/'+name+'_'+str(datetime.datetime.now()).split()[0]+'_'+str(args.task_id)+'.csv' for name in ['f','f1','f2','t','Jmax','DelS','tau','data']]
f.to_csv(filenames[0],index=False,header=False)
f1.to_csv(filenames[1],index=False,header=False)
f2.to_csv(filenames[2],index=False,header=False)
t.to_csv(filenames[3],index=False,header=False)
Jmax.to_csv(filenames[4],index=False,header=False)
DelS.to_csv(filenames[5],index=False,header=False)
tau.to_csv(filenames[6],index=False,header=False)
pd.DataFrame(data).to_csv(filenames[7])
